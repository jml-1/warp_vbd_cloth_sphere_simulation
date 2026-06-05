# -*-coding: utf-8 -*-

import argparse #用来解析命令行参数
from dataclasses import dataclass #用来解方便定义类中参数
from pathlib import Path #方便文件路径管理

import numpy as np
import warp as wp

MAX_NEIGHBOURS = 16#顶点可能在最大相邻顶点数
NUM_COLORS = 9 #3×3顶点着色
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "vbd_cloth_output_without_contact"

def vertex_index(ix:int,iy:int,nx:int)->int:
   #将二维网格顶点编号映射为一维顶点编号
   return ix + iy*nx

#构建邻接表，并在网格点间加入无向弹簧
#构建弹簧布料的基础函数
def add_spring(
      point_a:int,
      point_b:int,
      inital_length:float,#弹簧原长：即初始边长
      k:float,#弹簧边的刚度
      neighbours:np.ndarray,#点a的邻点b,点b的邻点a,相互加入：邻居顶点编号数组
      rest_lengths:np.ndarray,# 保存每个邻接弹簧的原长
      stiffness:np.ndarray,#保存每条边的刚度
      counts:np.ndarray,#每个顶点当前已经有多少个邻居
):
   """
   邻接表中添加无向弹簧，相互影响

   顶点间真正的拓扑关系表

   #遍历俩次,第一次: src = point_a, dst = point_b
   第二次: src = point_b,  dst = point_a
   """
   for src, dst in ((point_a, point_b),(point_b,point_a)):
      slot = counts[src]
      if slot >=MAX_NEIGHBOURS:
         raise RuntimeError("请增大 最大邻居个数：MAX_NEIGHBOURS ")
      neighbours[src,slot] = dst  #顶点src 的第slot 个邻点 是 dst
      rest_lengths[src,slot] = inital_length #顶点src 与第slot 个邻点的初始边长是 inital_length
      stiffness[src,slot] = k #顶点src 与第slot 个邻点的刚度是k
      counts[src] +=1 #顶点src保存的邻居点数+1
  

@dataclass#简化只用于保存数据的类的写法
class ClothData:
   '''
   CPU中的基础数据类
   这些基础数据类要上传到GPU设备端：使用warp来上传
   '''
  
   positions: np.ndarray         #顶点的初始位置:同时也是位移的参考坐标位置：位移的大小是与初始位置比较的结果
   
   pinned_positions:np.ndarray   #固定点位置：非固定的也填入初始位置，统一上传
   
   fixed : np.ndarray             #fiexed[i]  = 1,表示该顶点被固定，fixed[i] = 0,表示该顶点是自由点
   colors:np.ndarray              #vbd 并行着色编号，9种颜色，记为：0~8
   neighbours:np.ndarray          #记录顶点的相邻点：
   rest_lengths:np.ndarray        #弹簧边的初始长度
   stiffness:np.ndarray           #弹簧边的刚度
   faces:np.ndarray               #三角面片索引：只用于输出网格
   vertex_mass:float              #顶点的质量，用于 inertia = m / h^2
   dx :float                      #顶点间距，用于弹簧静止长度和牛顿步长限制


def build_square_cloth(
      resolution: int,     #方形布的边上布置的顶点数
      size: float,         #方形布的边长
      density: float,      #面的密度，用于计算质量 m = density * S
      stretch_stiffness:float,
      shear_stiffness:float,
      bend_stiffness:float,
)->ClothData:
   if resolution <3:
      raise ValueError("resolution must be at least 3")
   nx = resolution
   ny = resolution
   point_total = nx * ny
   dx = size / float(resolution - 1)

   position = np.zeros((point_total,3),dtype=np.float32)    #创建顶点容器，二维数组存放点坐标
   init_pinned_positions = np.zeros_like(position)          #保存固定顶点的固定位置，后续可对部分点强制恢复为该坐标位置
   fixed = np.zeros(point_total,dtype= np.int32)            #一维数组，用来标记每个顶点是否固定
   colors = np.zeros(point_total,dtype=np.int32)            #记录每个顶点颜色

   #生成规则方形网格顶点
   # x 横向覆盖[-size/2, size/2]，y从 0 向下排布
   for iy in range (ny):
      for  ix in range(nx):
         idx = vertex_index(ix,iy,nx)
         x = (float(ix) / float(nx-1) - 0.5) * size
         y = -float(iy) *dx
         z = 0.008 *dx * np.sin(1.73* ix + 2.41 * iy)  #z方向初始扰动
         position[idx] = (x,y,z)
         init_pinned_positions[idx] =position[idx]
         colors[idx] = (ix % 3) +3 *(iy%3)


   #生成布料网格顶点
   #左右两个点 fixed = 1,其余顶点在重力作用下自然下垂
   top_left = vertex_index(0, 0, nx)
   top_right = vertex_index(nx-1, 0, nx)
   fixed[top_left] = 1
   fixed[top_right] = 1
   position[top_right, 2] = 0.0 #该点zz坐标置为0
   position[top_left, 2] = 0.0  #该点zz坐标置为0
   init_pinned_positions[top_right] = position[top_right]      #将上述固定点位置保存到这里面
   init_pinned_positions[top_left] = position[top_left]

   neighbours = np.full((point_total,MAX_NEIGHBOURS),-1,dtype=np.int32)       # neighbours[3,5] = 7, 表示第3个顶点的第5个邻居点的编号是 7
   rest_lengths = np.zeros((point_total,MAX_NEIGHBOURS),dtype=np.float32)       #顶点与其相邻点的距离
   stiffness = np.zeros((point_total,MAX_NEIGHBOURS),dtype=np.float32)          #顶点与其相邻点的链接刚度
   num_neighbours = np.zeros(point_total,dtype=np.int32)                      #一维数组，记录每个顶点的邻点个数

   #为网格加入弹簧
   #1. 结构弹簧
   for iy in range(ny):
      for ix in range(nx):
         i = vertex_index(ix,iy,nx)
         if ix +1 < nx:
            add_spring(i,vertex_index(ix+1,iy,nx),dx,stretch_stiffness, neighbours, rest_lengths, stiffness, num_neighbours)
         if iy + 1 < ny:
            add_spring(i, vertex_index(ix, iy + 1, nx), dx, stretch_stiffness, neighbours, rest_lengths, stiffness, num_neighbours)
         if ix + 1 < nx and iy + 1 < ny:
            add_spring(i, vertex_index(ix + 1, iy + 1, nx), np.sqrt(2.0) * dx, shear_stiffness, neighbours, rest_lengths, stiffness, num_neighbours)
         if ix + 1 < nx and iy - 1 >= 0:
            add_spring(i, vertex_index(ix + 1, iy - 1, nx), np.sqrt(2.0) * dx, shear_stiffness, neighbours, rest_lengths, stiffness, num_neighbours)
         if ix + 2 < nx:
            add_spring(i, vertex_index(ix + 2, iy, nx), 2.0 * dx, bend_stiffness, neighbours, rest_lengths, stiffness, num_neighbours)
         if iy + 2 < ny:
            add_spring(i, vertex_index(ix, iy + 2, nx), 2.0 * dx, bend_stiffness, neighbours, rest_lengths, stiffness, num_neighbours)

    #每个四边形拆分为两个三角面片
   faces =[]
   for iy in range(ny-1):
      for ix in range(nx-1):
         a = vertex_index(ix,iy,nx)
         b = vertex_index(ix + 1, iy, nx)
         c = vertex_index(ix, iy + 1, nx)
         d = vertex_index(ix + 1, iy + 1, nx)
         faces.append((a,c,b))
         faces.append((b,c,d))
        
   area = size *size
   vertex_mass = density *area /float(point_total)
   
   return ClothData(
        positions=position,
        pinned_positions=init_pinned_positions,
        fixed=fixed,
        colors=colors,
        neighbours=neighbours.reshape(-1),       #将二维数组拉平为一维数组，方便在GPU中并行 二维访问：neighbours[i, k]
                                                #一维访问：    neighbours[i * MAX_NEIGHBOURS + k]
        rest_lengths=rest_lengths.reshape(-1),  #将二维数组拉平为一维数组，方便在GPU中并行
        stiffness=stiffness.reshape(-1),        #将二维数组拉平为一维数组，方便在GPU中并行
        faces=np.asarray(faces, dtype=np.int32),
        vertex_mass=vertex_mass,
        dx=dx,
    )

@wp.kernel
def predict_kernel(
   x: wp.array(dtype=wp.vec3),
   x_old: wp.array(dtype = wp.vec3),
   v: wp.array(dtype=wp.vec3),
   inertial: wp.array(dtype=wp.vec3),
   fixed: wp.array(dtype=wp.int32),
   pinned_x: wp.array(dtype=wp.vec3),
   gravity: wp.vec3,
   dt: float,
):
   tid = wp.tid() #每个block中的子线程
   if fixed[tid] != 0:  #该顶点非自由点，需要位置固定，速度清零
      p = pinned_x[tid]
      x[tid] = p
      x_old[tid] = p
      inertial[tid] = p
      v[tid] = wp.vec3(0.0,0.0,0.0)
   else:
      old = x[tid]
      x_old[tid] = old
      #关注点：速度的导数是加速度g,位移的导数是速度，通过一阶欧拉离散即可得到y
      y = old + v[tid] * dt + gravity * (dt *dt) 
      inertial[tid] = y
      x[tid] = y

#每个线程负责一个顶点的3×3局部问题：局部隐式能量最小化

@wp.kernel
def vbd_color_kernel(
   x: wp.array(dtype=wp.vec3),
   inertial: wp.array(dtype=wp.vec3),
   neighbors: wp.array(dtype=wp.int32),
   rest_lengths: wp.array(dtype=wp.float32),
   stiffness: wp.array(dtype=wp.float32),
   fixed: wp.array(dtype=wp.int32),
   colors: wp.array(dtype=wp.int32),
   active_color: int,
   inertia: float,
   max_step: float,
):
   tid = wp.tid()#获取当前block线程编号,一个线程负责一个顶点

   if fixed[tid] != 0 or colors[tid] != active_color:#按同一颜色顶点线程并行处理
      return

   xi = x[tid]
   yi = inertial[tid]

   # 梯度初始化为惯性项：inertia * (x_i - y_i)。
   gx = inertia * (xi[0] - yi[0])
   gy = inertia * (xi[1] - yi[1])
   gz = inertia * (xi[2] - yi[2])
    # 惯性项的Hessian 初始化为惯性项的 3x3 对角矩阵。
    # 为了减少寄存器和计算量，这里只存对称矩阵的 6 个独立元素：
    # [h00 h01 h02]
    # [h01 h11 h12]
    # [h02 h12 h22]
   h00 = inertia
   h01 = 0.0
   h02 = 0.0
   h11 = inertia
   h12 = 0.0
   h22 = inertia

   #遍历当前顶点的所有邻居弹簧，每个邻边都是一个弹簧，
   # 每条邻边能量 = 惯性能 + 弹簧能
   base = tid * MAX_NEIGHBOURS
   for slot in range(MAX_NEIGHBOURS):
        n = neighbors[base + slot] #因为邻接表被拉平为一维数组，所以tid个顶点的第slot个邻居为neighbors[tid * MAX_NEIGHBOURS + slot]

        if n >= 0:
            # 从扁平邻接表中读取一根弹簧：(tid, n)。
            # rest 是静止长度 L，k 是该弹簧的刚度。
            xj = x[n]
            rest = rest_lengths[base + slot]
            k = stiffness[base + slot]

            dx = xi - xj
            r = wp.length(dx)

            if r > 1.0e-7:
                # 单位方向 n = (x_i - x_j) / ||x_i - x_j||。
                inv_r = 1.0 / r
                nx = dx[0] * inv_r
                ny = dx[1] * inv_r
                nz = dx[2] * inv_r

                # 对弹簧能量 0.5*k*(r-L)^2 求梯度：
                # grad = k * (1 - L/r) * (x_i - x_j)。
                #当前长度 r 大于原长 rest，弹簧被拉伸，stretch > 0；如果 r < rest，弹簧被压缩，stretch < 0
                stretch = 1.0 - rest * inv_r

                gx += k * stretch * dx[0]
                gy += k * stretch * dx[1]
                gz += k * stretch * dx[2]

                # 弹簧 Hessian 的完整形式包含切向和法向分量。
                # 当弹簧被压缩时，精确 Hessian 可能非正定，导致牛顿步不稳定；
                # 这里把切向系数裁到非负，得到一个更稳的半正定近似。
                tangent = stretch
                if tangent < 0.0:
                    tangent = 0.0

                kt = k * tangent
                kn = k * (1.0 - tangent)

                # H = kt * I + kn * n*n^T，只累加对称矩阵的 6 个元素。
                h00 += kt + kn * nx * nx
                h01 += kn * nx * ny
                h02 += kn * nx * nz
                h11 += kt + kn * ny * ny
                h12 += kn * ny * nz
                h22 += kt + kn * nz * nz

   a = h00
   b = h01
   c = h02
   d = h11
   e = h12
   f = h22

    # 手写 3x3 对称矩阵求逆的伴随矩阵部分。
    # Warp kernel 中避免调用通用线性代数库，直接展开可以减少开销。
   cof00 = d * f - e * e
   cof01 = c * e - b * f
   cof02 = b * e - c * d
   cof11 = a * f - c * c
   cof12 = b * c - a * e
   cof22 = a * d - b * b
   
   #行列式
   det = a * cof00 + b * cof01 + c * cof02

   if det > 1.0e-10 or det < -1.0e-10:
        inv_det = 1.0 / det

        # 解 H * s = grad，局部牛顿更新为增量位移。
        sx = (cof00 * gx + cof01 * gy + cof02 * gz) * inv_det
        sy = (cof01 * gx + cof11 * gy + cof12 * gz) * inv_det
        sz = (cof02 * gx + cof12 * gy + cof22 * gz) * inv_det

        step = wp.vec3(-sx, -sy, -sz)  #求解得到该步的增量位移！！！！
        step_len = wp.length(step)

        if step_len > max_step:
            # 限制单次局部更新长度，避免低迭代数或极端参数下出现过冲。
            step = step * (max_step / step_len)

        x[tid] = xi + step   #更新位移！！！


# 收尾 kernel：用最终位置回算速度。
# VBD 求出的 x 是隐式积分后的新位置；速度用 (x_{n+1}-x_n)/h 得到，
# 再乘一个轻微阻尼，减少布料长时间振荡。
@wp.kernel
def finalize_kernel(
    x: wp.array(dtype=wp.vec3),
    x_old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    fixed: wp.array(dtype=wp.int32),
    pinned_x: wp.array(dtype=wp.vec3),
    dt: float,
    damping: float,
):
    tid = wp.tid()

    if fixed[tid] != 0:
        # 固定点速度始终为 0，位置始终等于钉住位置。
        p = pinned_x[tid]
        x[tid] = p
        v[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        v[tid] = ((x[tid] - x_old[tid]) / dt) * damping




def _format_float_array(values: np.ndarray, components: int = 1) -> str:
    """把浮点数组格式化成 VTK XML ASCII DataArray 需要的文本。"""

    flat = np.asarray(values, dtype=np.float32).reshape(-1, components)
    return "\n".join(" ".join(f"{x:.7g}" for x in row) for row in flat)


def _format_int_array(values: np.ndarray, components: int = 1) -> str:
    """把整数数组格式化成 VTK XML ASCII DataArray 需要的文本。"""

    flat = np.asarray(values, dtype=np.int32).reshape(-1, components)
    return "\n".join(" ".join(str(int(x)) for x in row) for row in flat)


def write_vtp(
    path: Path,
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_positions: np.ndarray,
    faces: np.ndarray,
    fixed: np.ndarray,
    colors: np.ndarray,
):
    """写出一帧 VTP PolyData，供 ParaView 读取。

    VTP 是 VTK XML PolyData 文件。这里把布料保存为三角面片网格，并额外写入
    点数据数组：
    - velocity：顶点速度向量。
    - speed：速度长度，方便在 ParaView 里直接按速度着色。
    - displacement：相对初始形状的位移。
    - fixed：是否为固定顶点。
    - vbd_color：VBD 的 9 色并行分组编号。

    文件使用 ASCII 格式，体积比二进制大一点，但便于调试和查看。
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    n_points = positions.shape[0]
    n_faces = faces.shape[0]
    displacement = positions - rest_positions
    speed = np.linalg.norm(velocities, axis=1).astype(np.float32)
    # VTK 的 Polys 由 connectivity 和 offsets 两个数组描述：
    # connectivity 是所有三角形顶点索引顺序拼接；
    # offsets 表示每个单元在 connectivity 中结束的位置。三角形每个单元 3 个点。
    offsets = np.arange(3, 3 * n_faces + 1, 3, dtype=np.int32)

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(f'    <Piece NumberOfPoints="{n_points}" NumberOfPolys="{n_faces}">\n')
        f.write('      <PointData Scalars="speed" Vectors="velocity">\n')
        f.write('        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">\n')
        f.write(_format_float_array(velocities, 3))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Float32" Name="speed" format="ascii">\n')
        f.write(_format_float_array(speed))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Float32" Name="displacement" NumberOfComponents="3" format="ascii">\n')
        f.write(_format_float_array(displacement, 3))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="fixed" format="ascii">\n')
        f.write(_format_int_array(fixed))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="vbd_color" format="ascii">\n')
        f.write(_format_int_array(colors))
        f.write("\n        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("      <CellData>\n")
        f.write("      </CellData>\n")
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii">\n')
        f.write(_format_float_array(positions, 3))
        f.write("\n        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Polys>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write(_format_int_array(faces.reshape(-1)))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write(_format_int_array(offsets))
        f.write("\n        </DataArray>\n")
        f.write("      </Polys>\n")
        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")


def write_pvd(path: Path, frames: list[tuple[int, float, str]]):
    """写出 ParaView 时间序列索引文件，指向 VTP 帧。"""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for _frame, time, filename in frames:
            f.write(f'    <DataSet timestep="{time:.7g}" group="" part="0" file="{filename}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")



def simulate(args):
   """
   布料仿真主流程
   循环结构为：
   1.构建CPU端布料数据
   2.上传CPU数据到GPU设备端：利用warp工具
   3.每帧执行一定的时间步
   4.每个时间步进行：位移预测、VBD颜色迭代和速度回算
   5.输出时间序列结果，paraview,查看
   """

   wp.init()

   #先在CPU端建立 方形布料网格，弹簧邻接表，固定点和三角面片
   cloth = build_square_cloth(
        resolution = args.resolution,
        size = args.size,
        density= args.density,
        stretch_stiffness= args.stretch_stiffness,
        shear_stiffness=args.shear_stiffness,
        bend_stiffness=args.bend_stiffness,
   )

   device = args.device #确定用户指定的GPU运行设备，即零号显卡上进行cuda 运行
   n_vertices = cloth.positions.shape[0] #取二维数组positions的零维大小，即顶点数

   #将CPU 基础数据(数据格式：numpy 数组 )上传到GPU设备端
   x =wp.from_numpy(cloth.positions,dtype=wp.vec3,device=device)     #当前顶点位置
   x_old= wp.from_numpy(cloth.positions.copy(),dtype=wp.vec3,device=device)#上一时间步位置
   v= wp.zeros(n_vertices, dtype=wp.vec3,device=device)              #顶点速度，初始为零
   inertial =wp.from_numpy(cloth.positions.copy(),dtype=wp.vec3,device=device)#保存预测位置，在VBD惯性项中使用它

   #固定顶点、颜色、顶点间弹簧拓扑结构在运动过程中不发生变化，只需上传一次即可
   pinned_x = wp.from_numpy(cloth.pinned_positions,dtype=wp.vec3,device=device)
   fixed = wp.from_numpy(cloth.fixed,dtype=wp.int32,device=device)
   color = wp.from_numpy(cloth.colors,dtype=wp.int32,device=device)
   neighbours = wp.from_numpy(cloth.neighbours,dtype=wp.int32,device=device)#neighbours已经被转化为一维数组了
   rest_lengths = wp.from_numpy(cloth.rest_lengths,dtype=wp.float32,device=device)
   stiffness = wp.from_numpy(cloth.stiffness,dtype=wp.float32,device=device)        #这里的刚度已经包括了剪切或弯曲刚度
   colors = wp.from_numpy(cloth.colors, dtype=wp.int32, device=device)
   
   out_dir = OUTPUT_DIR

   frame_dt = 1.0 /args.fps  #输出帧时间间隔
   sub_dt = frame_dt / float(args.substeps)#求解的子书简不间隔
   inertia = cloth.vertex_mass / (sub_dt * sub_dt) #隐式欧拉目标函数中的质量权重
   max_step = args.max_step_scale * cloth.dx #质点单步最大位移
   gravity = wp.vec3(0.0,args.gravity,0.0) #定义y方向重力
   saved_vtk_frames: list[tuple[int, float, str]] = []

   # VTP 输出从第 0 帧开始记录。
   zero_velocity = np.zeros_like(cloth.positions)
   write_vtp(
      out_dir / "frame_0000.vtp",
      cloth.positions,
      zero_velocity,
      cloth.positions,
      cloth.faces,
      cloth.fixed,
      cloth.colors,
   )
   saved_vtk_frames.append((0, 0.0, "frame_0000.vtp"))


   for fram in range(1,args.frames +1):
      for _substep in range(args.substeps):
         #预测步：计算隐式欧拉的惯性位置y：显示计算得到
         wp.launch(
            predict_kernel,
            dim = n_vertices,
            inputs = [x,x_old,v,inertial,fixed,pinned_x,gravity,sub_dt],
            device=device,
         )

         #VBD迭代：每次迭代一次扫描9个颜色
         #同一个颜色顶点间没有直接弹簧链接，因此该颜色内可以GPU并行更新
         for _iter in range(args.vbd_iters):
            for color in range(NUM_COLORS):
               wp.launch(
                  vbd_color_kernel,
                  dim = n_vertices,
                  inputs=[
                        x,
                        inertial,
                        neighbours,
                        rest_lengths,
                        stiffness,
                        fixed,
                        colors,
                        color,      #按同一颜色顶点并行处理
                        inertia,
                        max_step,
                  ],
                  device=device,
               )
               
         # 用本子步开始和结束的位置差回算速度，并施加阻尼。
         wp.launch(
               finalize_kernel,
               dim=n_vertices,
               inputs=[x, x_old, v, fixed, pinned_x, sub_dt, args.damping],
               device=device,
         )

         if fram % args.save_every == 0 or fram == args.frames:
            # 只有保存输出时才把 GPU 位置/速度拷回 CPU，减少不必要的数据传输。
            positions = x.numpy()
            velocities = v.numpy()

            write_vtp(
                out_dir / f"frame_{fram:04d}.vtp",
                positions,
                velocities,
                cloth.positions,
                cloth.faces,
                cloth.fixed,
                cloth.colors,
            )
            saved_vtk_frames.append((fram, fram * frame_dt, f"frame_{fram:04d}.vtp"))

            print(f"saved frame {fram:04d}")

   pvd_path = out_dir / "cloth.pvd"
   write_pvd(pvd_path, saved_vtk_frames)
   print(f"wrote ParaView VTP frames to: {out_dir.resolve()}")
   print(f"open this file in ParaView for the cloth time series: {pvd_path.resolve()}")

def parse_args():
   '''
   命令行参数

   分辨率、帧数、刚度、密度，输出参数等
   '''
   parser = argparse.ArgumentParser(
      description= "采用VBD与Warp方式, 在GPU中进行方形布料模拟."
   )
   parser.add_argument("--resolution", type= int, default=41,help="方形布料边上不知41个点")#总顶点数 = 41 ×41
   parser.add_argument("--size", type=float, default=2.0, help="方形布料的边长")
   parser.add_argument("--density", type=float, default=0.18, help="布料密度")
   parser.add_argument("--stretch-stiffness", type=float, default=1500.0, help="弹簧的刚度")
   parser.add_argument("--shear-stiffness", type=float, default=1000.0, help="弹簧剪切刚度")
   parser.add_argument("--bend-stiffness", type=float, default=90.0, help="弹簧弯曲刚度")

   parser.add_argument("--vbd-iters", type=int, default=18, help="每个子时间步内进行多少次VBD迭代")
   parser.add_argument("--damping", type=float, default=0.992, help="阻尼系数")
   parser.add_argument("--gravity", type=float, default=-9.81, help="Y方向的重力加速度")
   parser.add_argument("--max-step-scale", type=float, default=0.6, help="单步内最大允许移动的位移系数, 即: 该系数乘以dx")
   
   parser.add_argument("--device", default="cuda:0", help="指定Warp 在Gpu设备中运行")

   parser.add_argument("--frames", type=int, default=40, help="模拟输出的帧数")
   parser.add_argument("--fps", type=float, default=60.0, help="模拟帧率")
   parser.add_argument("--substeps", type=int, default=1, help="每帧的子步数")

   parser.add_argument("--out-dir", default=str(OUTPUT_DIR), help="VTP格式目录固定为脚本所在目录下的 vbd_cloth_output")
   parser.add_argument("--save-every", type=int, default=5, help="VTP save interval in frames")

   return parser.parse_args()

if __name__ == "__main__":
   simulate(parse_args())

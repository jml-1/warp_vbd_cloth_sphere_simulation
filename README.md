# VBD Cloth Simulation

This is a NVIDIA Warp GPU cloth simulation demo. The top-left and top-right
corners of a square cloth are pinned, while the rest of the cloth falls under
gravity. The solver uses a Vertex Block Descent style local Newton update with
9-color parallel sweeps.

The current version also includes a dynamic rigid sphere collider. Cloth-sphere
contacts are generated on the GPU, compacted with a prefix-sum stream compaction
pass, and solved as position constraints with two-way coupling between cloth
vertices and the sphere center.

## Install Dependencies

Run this in the VSCode terminal:

```powershell
python -m pip install -r requirements.txt
```

Check that Warp can see the GPU:

```powershell
python -c "import warp as wp; wp.init(); print(wp.is_cuda_available()); print(wp.get_device('cuda:0'))"
```

## Run In VSCode

1. Open this folder in VSCode.
2. Open `Run and Debug`.
3. Select `Run GPU VBD Cloth`.
4. Click the green run button.

The default run writes:

```text
output/combined_scene.pvd
output/scene_0000.vtp
output/scene.pvd
output/cloth.pvd
output/sphere.pvd
output/frame_0000.vtp
output/sphere_0000.vtp
output/frame_0001.vtp
...
output/frame_0240.vtp
```

Open `output/combined_scene.pvd` in ParaView and click `Apply`. This is the most
reliable view because each `scene_XXXX.vtp` contains both the cloth mesh and the
rigid sphere mesh. You can color by `object_id`: `0` is cloth and `1` is sphere.

`output/scene.pvd` is also written as a two-part collection, while `cloth.pvd`
and `sphere.pvd` are separate single-object time series. The cloth VTK files
include these point arrays: `velocity`, `speed`, `displacement`, `fixed`,
`vbd_color`, and `sphere_contact`.

To also write OBJ files, run the VSCode configuration
`Run GPU VBD Cloth with OBJ`.

## Command Line

```powershell
python warp_vbd_cloth.py --device cuda:0 --frames 240 --substeps 4 --vbd-iters 40 --contact-iters 8 --stretch-stiffness 2500 --shear-stiffness 900 --bend-stiffness 25 --damping 0.985 --sphere-radius 0.35 --sphere-position 0.0 -1.35 0.0 --sphere-mass 8.0 --save-every 1 --out-dir output --no-obj
```

Useful sphere/contact parameters:

```text
--sphere-radius          sphere collision radius
--sphere-position X Y Z  initial sphere center
--sphere-mass            sphere mass; 0 makes it effectively static
--sphere-gravity-scale   0 by default, 1 makes the sphere fall under gravity
--contact-iters          collision projection iterations per substep
--contact-relaxation     collision constraint projection strength
--cloth-thickness        extra cloth collision thickness
--contact-margin         near-contact generation margin
```


'''
在pdm环境下的运行命令
布料自然下垂案例，运行命令：pdm run cloth
布料与小球接触（双向耦合）案例，运行命令：pdm run cloth_contact

结果查看方式：
1：布料自然下垂案例
结果存放在：vbd_cloth_output_without_contact 文件夹中
将文件： cloth.pvd 直接托入paraview 软件中点击时间运行按钮即可显示效果 

2：布料与小球接触（双向耦合）案例 
结果存放在：vbd_cloth_output_with_contact 文件夹中
将文件： scene.pvd 直接托入paraview 软件中点击时间运行按钮即可显示效果 

'''

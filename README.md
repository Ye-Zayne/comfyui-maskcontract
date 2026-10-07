# MaskContract · 蒙版输入契约检查

在 inpaint 或 masked composite 前检查 IMAGE + MASK：抓出尺寸不一致、隐式批次广播、NaN/无穷值、越界数值，以及空白/整张蒙版。输出每帧覆盖率、像素 bbox、叠加证据、JSON 和判定，也可用 Gate 阻止不符合预设条件的输入继续执行。

[English documentation](README.en.md)

## 安装

将仓库放到 `ComfyUI/custom_nodes/comfyui-maskcontract` 并重启 ComfyUI；正式注册后也可通过 Manager 安装。使用 ComfyUI 已有的 torch、NumPy 和 Pillow 环境，无模型、网络服务、密钥或运行时文件写入。版本 `0.1.0`，作者 A ad钙，MIT。

## 节点

| 节点 | 输出 | 用法 |
| --- | --- | --- |
| `MaskContractInspect` | overlay IMAGE、report STRING、verdict INT | 独立 OUTPUT_NODE；可把 overlay 接 PreviewImage |
| `MaskContractGate` | 原 IMAGE、验证后的 MASK、report STRING、verdict INT | 把输出接实际编辑节点；默认阻止 FAIL 和 REVIEW |

`0=FAIL`、`1=REVIEW`、`2=PASS`。形状/批次/数值/覆盖范围错误为 FAIL；阈值选中空区域或整个画布，默认 REVIEW。`extreme_policy` 可设为 fail 或 allow，适合明确要保留空蒙版/全图操作的流程。Gate 的 `allow_review=true` 明确允许 REVIEW，FAIL 始终阻止。Inspect 和 Gate 都是 OUTPUT_NODE，报告节点不会因没有下游连接被跳过；前端文字渲染取决于 ComfyUI 版本，STRING 输出始终可读取。

Gate 必须接入要保护的下游依赖链：编辑节点应读取 Gate 的 IMAGE/MASK 输出。仅添加一个断开的检查分支，不能保证在其它独立采样分支运行前阻断它们；Inspect 本身也不会阻断执行。

## 严格约定

- IMAGE 必须为浮点 `[B,H,W,3或4]`；MASK 为浮点 `[B,H,W]`。不自动把 2D 或通道式蒙版改成 MASK。
- 输入必须有限且处于 `[0,1]`，软蒙版可用；不缩放、裁切、反相、归一化或 clip。
- MASK 的 H/W 必须对应所检查 IMAGE 的**像素画布**，不是 VAE latent 的 H/8、裁剪区域或屏幕显示尺寸。合成时通常应检查 **source image** 与它的 mask；本节点不核验 destination 的位置/重叠或 VAE latent。
- B 必须相等。只有 `allow_batch1_broadcast=true` 才允许单 MASK 明确覆盖整批 IMAGE；Gate 此时返回扩展后的 `[B,H,W]` view，保留原数值、dtype/device。不循环重复多张 MASK。
- `white_convention` 只是你声明的用途：白色表示编辑/保护/选用 source；**不自动判断极性或语义正确性**。意外反相可通过覆盖率与叠加观察、或已配置的覆盖上限抓出，但不能凭图像知道你真正想编辑哪里。

`coverage` 是 `mask >= active_threshold` 的像素比例，默认阈值 0.5；`mean_mask_value` 是软蒙版均值，两者不同。`min_coverage`/`max_coverage` 为包含端点的允许范围，每帧检查。bbox `[x1,y1,x2,y2]` 的右/下边界不包含；没有选中像素时为 null。`all_empty/all_full` 分别表示所有值精确为 0/1，另有阈值选中 none/all 的字段。

叠加只做证据：选中白值以青色（保护约定为紫色）按软值染色，帧边框绿/黄/红表示 PASS/REVIEW/FAIL；RGBA 的 alpha 保留。有效 IMAGE 配错误 MASK 时只显示原图及红边框；IMAGE 无效或资源超限时返回标明原因的 32×32 红色占位，不伪造输入画布。不会把 overlay 作为 Gate 的编辑输入。

资源上限在值扫描/传输/叠加分配前检查：默认 64 帧、全批共 8,388,608 像素；可配置到最多 256 帧、16,777,216 像素。超限为 FAIL，需拆分批次或明确调整预算。Inspect 的叠加是 CPU float32；Gate 保留原输入，除显式 batch-1 扩展。检查只能保证当前张量的数值契约，不覆盖任意并发修改或后续节点的行为。

## 与已有节点的区别

ComfyUI 官方已有 [MaskComposite](https://docs.comfy.org/built-in-nodes/MaskComposite)、InvertMask、MaskToImage 等操作；[官方 ImageCompositeMasked 实现](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_mask.py) 会为合成插值蒙版并重复批次。MaskContract 负责在这些操作之前检查明确的尺寸、批次和覆盖约定，不替代这些变换节点，也不宣称没有类似工具。[EditFence](https://github.com/Ye-Zayne/comfyui-editfence) 对比原图与编辑结果，检查允许区域外是否被改动；本包仅检查编辑**输入**，不对结果做保真验收。

## 示例

把 `examples/maskcontract_image.png`、`maskcontract_alpha_mask.png`、`maskcontract_background.png` 拷到 ComfyUI input 目录。`inspect_api.json` 把 Inspect overlay 接 PreviewImage；`gate_composite_api.json` 把 Gate 的 IMAGE/MASK 接官方 ImageCompositeMasked 并预览结果。蒙版 PNG 在 alpha 中存 `1-mask`，对应 LoadImage MASK；普通白色 RGB PNG 没有 alpha 时不等于白色 MASK。

两个 JSON 都是 **API prompt**，通过 `POST /prompt` 发送 `{"prompt": <JSON内容>}`，不是可直接导入画布的 canvas workflow。示例预期覆盖率 0.1875、bbox `[48,32,120,96]`。另附 64×64 的 `maskcontract_wrong_canvas.png`：替换 mask 图后 Inspect 应报 FAIL；Gate 会阻断。无需执行生成脚本即可使用已附 PNG。

## 开发验证

```sh
python -m pip install pytest ruff torch numpy Pillow
python -m pytest --rootdir=tests --confcutdir=tests tests --import-mode=importlib -q
python -m ruff check . --select S102,S307,E702
```

示例生成器仅用于开发，`.comfyignore` 排除 tests、CI 和生成脚本。运行节点不执行文件写入、动态导入、环境变量读取、数据库、命令行或网络访问。

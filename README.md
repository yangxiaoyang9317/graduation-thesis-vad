# 多源复杂环境下的语音端点检测算法研究

**毕业设计 | 基于深度学习的语音端点检测（VAD）**

---

## 项目简介

本项目为本科毕业设计，研究在多源复杂噪声环境下的语音端点检测（Voice Activity Detection, VAD）算法，基于 ICASSP 2021 论文 *Self-Attentive VAD: Context-Aware Detection of Voice from Noise* 进行复现与改进。

---

## 目录结构

```
graduation-thesis-vad/
├── thesis/                          # 论文文档
│   ├── 毕业论文_多源复杂环境下的语音端点检测算法研究.docx   # 完整毕业论文
│   ├── 开题报告_2023.pptx           # 开题报告 PPT
│   ├── 中期审查意见.docx             # 中期审查意见
│   ├── 中期总结.doc                  # 中期总结报告
│   ├── 基于深度学习的语音端点检测（阶段稿）.docx
│   └── references/                  # 参考文献
│       ├── ICASSP2021_Self-Attentive-VAD_Paper.pdf
│       └── ICASSP2021_Self-Attentive-VAD_Notes.docx
└── code/                            # 源代码
    ├── vad/                         # 核心模型代码
    ├── tests/                       # 测试用例
    ├── main.py
    ├── requirements.txt
    └── Dockerfile
```

---

## 核心论文

> **Self-Attentive VAD: Context-Aware Detection of Voice from Noise**
> Yong Rae Jo, Youngki Moon, Won Ik Cho, Geun Sik Jo
> ICASSP 2021 — [IEEE Link](https://ieeexplore.ieee.org/document/9413961)

---

## 模型架构

使用自注意力机制（Self-Attention）进行声学帧级别的语音/噪声分类，相比传统 RNN 方案：
- 更强的上下文感知能力
- 更低的计算复杂度
- 在低信噪比场景下表现更鲁棒

---

## 快速开始

### 环境安装

```bash
cd code
pip install -r requirements.txt
```

### 训练

```bash
python main.py train configs/your_config.yaml
```

### 评估

```bash
python main.py evaluate <eval_path> <checkpoint_path>
```

### 推理

```bash
python main.py predict <audio_path> <checkpoint_path>
```

---

## 引用

```bibtex
@INPROCEEDINGS{9413961,
  author={Jo, Yong Rae and Ki Moon, Young and Cho, Won Ik and Sik Jo, Geun},
  booktitle={ICASSP 2021},
  title={Self-Attentive VAD: Context-Aware Detection of Voice from Noise},
  year={2021},
  pages={6808-6812},
  doi={10.1109/ICASSP39728.2021.9413961}
}
```

---

*王靖洋 · 本科毕业设计 · 2023*

# 多源复杂环境下的语音端点检测算法研究

**本科毕业设计 · 王靖洋 · 2023**

基于自注意力机制（Self-Attention）的深度学习语音端点检测（VAD），
复现并扩展 ICASSP 2021 论文 *Self-Attentive VAD: Context-Aware Detection of Voice from Noise*，
重点研究多源复杂噪声环境下的检测鲁棒性。

---

## 研究背景

语音端点检测（Voice Activity Detection, VAD）是语音识别、语音增强、语音压缩等应用的基础预处理步骤，用于准确判断输入信号中语音段的起止位置。

在**多源复杂噪声环境**中（多个噪声源同时存在、SNR 低、噪声类型多变），传统方法存在明显局限：

| 方法类型 | 代表 | 问题 |
|----------|------|------|
| 信号处理 | 过零率、能量阈值 | 低 SNR 下失效 |
| 传统机器学习 | GMM-HMM | 特征工程依赖强，泛化差 |
| RNN/LSTM | bDNN, ACAM | 难以建模全局上下文，梯度消失 |
| **自注意力（本文）** | **Self-Attentive VAD** | **帧间上下文感知强，鲁棒性优** |

---

## 模型架构

### 总体流程

```
音频输入
  ↓ STFT → Log-Mel / MRCG 特征（25ms 窗、10ms 步长、80 mel bins）
  ↓ 上下文拼接（context window: w=19, u=9）
  ↓ 线性投影 → d_model=128
  ↓ 正弦位置编码（Sinusoidal Positional Encoding）
  ↓ Transformer Encoder × 3 层（单头自注意力 + FFN × 4）
  ↓ 帧级二分类（语音 / 非语音）
  ↓ Boosted 后处理（滑动平均平滑）
  ↓ VAD 标记输出
```

### SelfAttentiveVAD（主模型）

```
TransformerEncoder (×3 layers)
  └─ MultiHeadAttention (n_heads=1, d_model=128)
  └─ PositionwiseFeedForward (d_ff=512)
  └─ Pre-LN + Dropout + Residual
```

### 对比基线模型

| 模型 | 架构 | 特点 |
|------|------|------|
| **SelfAttentiveVAD**（本文） | 3 层 Transformer Encoder | 上下文感知，全局依赖 |
| bDNN（Boosted DNN） | 2 层 DNN + BN + 滑动平均 | 轻量，局部特征 |
| ACAM | LSTM + 自适应上下文注意力 | 窗口内注意，编解码结构 |

---

## 数据集

### 训练数据

| 数据集 | 说明 |
|--------|------|
| **TIMIT** | 6300 句，630 说话人，16kHz，美式英语，音素级标注 |
| MUSAN | 多类型噪声（音乐/噪声/语音）|
| MR | 网络爬取的乐器伴奏文件（.wav），人工筛选 |
| Sound Effect Library | 音效库 |
| YouTube-SoundEffect | YouTube 音效 |
| YouTube EditorSoundEffect | YouTube 编辑器音效 |

**噪声混合策略：**
- **MUSAN 场景**：MUSAN 噪声 + TIMIT 语音合成
- **MIX 场景**：四类噪声库随机采样叠加（模拟真实多源环境），每段噪声周期性插入 10% 静音
- 信噪比（SNR）：**-6 dB 至 +12 dB** 随机采样
- 每条 TIMIT 语音前后各填充 2 秒静音

### 测试数据

| 测试集 | 内容 | 特点 |
|--------|------|------|
| Noisex-TIMIT | TIMIT + Noisex-92 噪声 | 5 个 SNR 级别：-10/-5/0/5/10 dB |
| ENVIRONMENT | 公园/公交站/建筑工地/室内，移动设备录音 | 真实环境噪声 |
| MOVIE | 多部电影音轨，.srt 字幕辅助标注 | 复杂音频事件 |
| YOUTUBE | YouTube-8M 随机抽取，约 24 小时，人工标注 | 最复杂，真实世界 |

---

## 实验配置

| 超参数 | 值 |
|--------|-----|
| 声学特征 | Log-Mel Spectrogram（n_mels=80）+ MRCG |
| 帧长 / 帧移 | 25ms / 10ms |
| 上下文窗口 | half=19, jump=9, shift=39 帧 |
| d_model | 128 |
| Transformer 层数 | 3 |
| FFN 维度 | 512 |
| 注意力头数 | 1 |
| Dropout | 0.3 |
| 批量大小 | 4096 |
| 优化器 | Adam (lr=1e-4, weight_decay=1e-5) |
| 梯度裁剪 | [-1, 1] |
| 训练轮数 | 30 epochs |
| 模型选择 | 验证集 AUC 最高的 checkpoint |

---

## 数据增强

在原论文基础上，本实现新增了以下增强策略：

### SpecAugment（频谱增强）
对 Log-Mel 频谱图进行随机掩码，提升对频率和时间扰动的鲁棒性：

```python
SpecAugment(
    num_mask=2,
    freq_masking=0.15,   # 最大遮蔽 15% 频带
    time_masking=0.20,   # 最大遮蔽 20% 时间帧
)
```

### 语音-噪声动态混合
训练时对每条语音实时注入随机噪声（SNR 范围 -5 ~ +6 dB），
使模型在不同信噪比条件下均能有效训练，不依赖固定 SNR 的预生成数据集。

### BoostedDNN 输入 Dropout
原论文 BoostedDNN 无输入 Dropout，本实现在第一层前增加了 Dropout，
有效缓解在 MIX 多噪声场景下的过拟合。

---

## 实验结果

### Noisex-92 传统噪声场景（表 2）

在婴儿哭声（babbling）场景外，Self-Attentive VAD 在各类 Noisex-92 噪声下均优于 bDNN 和 ACAM，在**因子噪声（Factory 1/2）** 等高强度场景中优势尤为明显。

- ACAM 在 MUSAN 婴儿哭声场景略优（其自适应上下文注意力有助于分离类语音噪声）
- MIX 场景中 Self-Attentive VAD 和 bDNN 均超过 ACAM（ACAM 对背景噪声变化敏感）
- **低 SNR（≤ 0 dB）** 场景下，Self-Attentive VAD 的优势最为突出

### 真实世界音频（表 3）

| 数据集 | 结论 |
|--------|------|
| MOVIE | 除《阿甘正传》《律政俏佳人》外，均优于 ACAM；在《世界末日》《独立日》《拯救大兵瑞恩》等动作类高噪声影片中优势最大 |
| YOUTUBE | 显著优于 bDNN 和 ACAM，在真实世界复杂音频上扩展性最强 |
| ENVIRONMENT | 多场景（公园/公交/工地/室内）均有竞争力表现 |

### 模型参数对比

Self-Attentive VAD 的参数量**远小于** ACAM 和 bDNN，同时维持相当或更优的性能，具有更好的参数效率，适用于资源受限的嵌入式/移动端场景。

### TIMIT 基准

在 TIMIT 数据集上，F1 分数可达 **0.97 以上**，不同噪声环境下表现稳定。

---

## 关键发现

1. **自注意力 > RNN**：ACAM 基于 LSTM + 注意力的编解码结构在处理 VAD 逐帧分类时存在天然局限，Self-Attention 的并行全局感知在多源噪声下更有优势。

2. **MIX 场景 > MUSAN 场景**：随着噪声源多样性增加，Self-Attentive VAD 的相对优势显著提升，说明其上下文建模能力在复杂噪声下更能发挥作用。

3. **音频"词汇"类比**：自注意力机制令模型学习音频事件间的依赖关系（类似 NLP 中的词法关系），在复杂上下文中识别语音"token"，这一机制在多声源场景中尤为有效。

4. **低 SNR 鲁棒**：-10 dB 极端低信噪比条件下，Self-Attentive VAD 保持了比基线更平稳的性能曲线。

---

## 快速开始

### 安装

```bash
cd code
pip install -r requirements.txt
```

### 训练

```bash
python main.py train <config_path>
# 示例（使用测试配置）：
python main.py train tests/configs/vad/train_config.yaml
```

### 评估

```bash
python main.py evaluate <eval_path> <checkpoint_path>
```

### 推理（单文件）

```bash
python main.py predict <audio_path> <checkpoint_path>
```

### Docker

```bash
docker build -t vad .
docker run vad python main.py predict audio.wav checkpoint.pt
```

---

## 目录结构

```
graduation-thesis-vad/
├── thesis/
│   ├── 毕业论文_多源复杂环境下的语音端点检测算法研究.docx
│   ├── 开题报告_2023.pptx
│   ├── 中期审查意见.docx
│   ├── 中期总结.doc
│   ├── 基于深度学习的语音端点检测（阶段稿）.docx
│   └── references/
│       ├── ICASSP2021_Self-Attentive-VAD_Paper.pdf
│       └── ICASSP2021_Self-Attentive-VAD_Notes.docx
└── code/
    ├── vad/
    │   ├── models/           # SelfAttentiveVAD / BoostedDNN / ACAM / DNN
    │   ├── modeling/         # Transformer, Multi-Head Attention, 位置编码
    │   ├── acoustics/        # 特征提取, SpecAugment, 语音-噪声混合
    │   ├── datasets/         # 数据集加载
    │   ├── training/         # Trainer, Checkpointer, Logger
    │   ├── postprocessing/   # 后处理（平滑、切分）
    │   ├── metrics.py        # VAD Accuracy, EER, 边界精度
    │   └── losses.py         # Token NLL Loss
    ├── tests/
    ├── main.py
    ├── requirements.txt
    └── Dockerfile
```

---

## 评估指标

代码实现了论文中的 **VAD Accuracy（VACC）**，综合考虑帧级精度和边界精度：

```
VACC = harmonic_mean(
    Frame Accuracy (ACC),
    Start Boundary Accuracy (SBA),
    End Boundary Accuracy (EBA),
    Border Precision (BP)
)
```

其中 SBA / EBA 对边界附近的帧采用**距离加权**方案，BP 惩罚过多的预测分段数。

此外还实现了 **EER（Equal Error Rate）** 指标用于 ROC 分析。

---

## 引用

```bibtex
@INPROCEEDINGS{9413961,
  author={Jo, Yong Rae and Ki Moon, Young and Cho, Won Ik and Sik Jo, Geun},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech
             and Signal Processing (ICASSP)},
  title={Self-Attentive VAD: Context-Aware Detection of Voice from Noise},
  year={2021},
  pages={6808-6812},
  doi={10.1109/ICASSP39728.2021.9413961}
}
```

---

## 致谢

感谢毕设指导老师**钱宇华老师**的悉心指导，以及参考实现的原作者 [voithru/voice-activity-detection](https://github.com/voithru/voice-activity-detection)。

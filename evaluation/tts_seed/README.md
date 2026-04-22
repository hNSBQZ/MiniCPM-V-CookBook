# MiniCPM-o 4.5 C++ TTS 评测说明

`tts_seed/` 是一套面向 `seed-tts-eval` 中文测试集的 C++ 版 TTS 评测流水线。它做了三件事：

1. 用 `extract_prompt_bundle.py` 从参考音频提取 `prompt_bundle`
2. 用 `generate_cpp.py` 调用 `llama-omni-tts-eval` 批量生成 wav
3. 计算 WER 和说话人相似度（SIM / ASV）

和原始 `seed-tts-eval` 相比，这里已经把**评测脚本相关代码**复制到了本目录下的 `eval_tools/`，避免再依赖外部 `seed-tts-eval` 仓库里的那几份脚本。

## 目录说明

```text
tts_seed/
├── README.md
├── diff-master.patch              # llama.cpp-omni 的完整改动
├── run_tts_eval_cpp_zh.sh         # 完整流水线入口
├── run_eval_only.sh               # 只跑 WER + SIM
├── generate_cpp.py                # Python 调度层
├── extract_prompt_bundle.py       # prompt_bundle 提取
├── eval_tools/                    # 从 seed-tts-eval 本地化过来的评测脚本
│   ├── get_wav_res_ref_text.py
│   ├── run_wer.py
│   ├── average_wer.py
│   └── speaker_verification/
│       ├── verification_pair_list_v2.py
│       ├── verification.py
│       ├── average.py
│       └── models/
└── md/                            # 设计和过程文档
```

## 1. 先准备 llama.cpp-omni

`llama-omni-tts-eval` 不是上游原生自带能力，这里的 C++ 改动都在 `diff-master.patch` 里。

先 clone 上游 `llama.cpp-omni`，然后打补丁并编译：

```bash
git clone https://github.com/tc-mb/llama.cpp-omni.git
cd llama.cpp-omni
git apply /path/to/Video-MME/cpp-eval/tts_seed/diff-master.patch

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-omni-tts-eval -j"$(nproc)"
```

编译完成后应当能看到：

```bash
./build/bin/llama-omni-tts-eval --help
```

## 2. Python 环境

推荐单独准备一个 Python 环境，例如：

```bash
conda create -n minicpmo45_eval python=3.10 -y
conda activate minicpmo45_eval
```

安装本流水线涉及的依赖：

```bash
pip install tqdm numpy scipy soundfile librosa jiwer zhon zhconv \
    onnxruntime s3tokenizer funasr transformers sentencepiece \
    fire packaging
pip install torch torchaudio
```

说明：

- `extract_prompt_bundle.py` 需要 `torch`、`torchaudio`、`s3tokenizer`、`onnxruntime`
- `run_wer.py` 需要 `funasr`、`jiwer`、`zhconv`
- SIM 评测需要 `torch`、`torchaudio`、`librosa`

## 3. 代码依赖

### 已经复制到本目录的代码

下面这些原本来自 `seed-tts-eval` 的脚本，已经复制到了 `tts_seed/eval_tools/`：

- `get_wav_res_ref_text.py`
- `run_wer.py`
- `average_wer.py`
- `speaker_verification/verification_pair_list_v2.py`
- `speaker_verification/verification.py`
- `speaker_verification/average.py`
- `speaker_verification/models/ecapa_tdnn.py`

### 仍需额外准备的第三方代码

说话人相似度评测底层会通过 `torch.hub` 加载 `s3prl` 的 `wavlm_large` upstream。这个仓库体积较大，没有一并复制到这里，建议单独 clone：

```bash
cd /path/to/Video-MME/cpp-eval/tts_seed/eval_tools
git clone https://github.com/s3prl/s3prl.git s3prl-main
```

或者不放在默认位置，改用环境变量指定：

```bash
export S3PRL_REPO=/path/to/s3prl
```

## 4. 模型依赖下载

### 4.1 GGUF 主模型目录

最简单的做法是把整个 `openbmb/MiniCPM-o-4_5-gguf` 仓库下载到一个目录，避免缺少 `audio/tts/token2wav` 子模型。

```bash
huggingface-cli download openbmb/MiniCPM-o-4_5-gguf \
  --local-dir /path/to/MiniCPM-o-4_5-gguf
```

然后设置：

```bash
export MODEL_PATH=/path/to/MiniCPM-o-4_5-gguf
```

常见会用到的文件包括：

- `MiniCPM-o-4_5-F16.gguf` 或 `MiniCPM-o-4_5-Q4_K_M.gguf`
- `audio/MiniCPM-o-4_5-audio-F16.gguf`
- `tts/MiniCPM-o-4_5-projector-F16.gguf`
- `tts/MiniCPM-o-4_5-tts-F16.gguf` 或 `tts/MiniCPM-o-4_5-tts-step-audio-F16.gguf`
- `token2wav-gguf/*`

如果你只想测量化 TTS 子模型，也可以单独覆盖：

```bash
export TTS_MODEL_PATH=/path/to/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf
```

### 4.2 prompt_bundle 提取所需 ONNX 模型

`extract_prompt_bundle.py` 需要两个 ONNX 文件，来源是 Hugging Face 仓库 `stepfun-ai/Step-Audio-2-mini` 的 `token2wav/` 目录：

- `speech_tokenizer_v2_25hz.onnx`
- `campplus.onnx`

下载示例：

```bash
huggingface-cli download stepfun-ai/Step-Audio-2-mini \
  token2wav/speech_tokenizer_v2_25hz.onnx \
  token2wav/campplus.onnx \
  --local-dir /path/to/Step-Audio-2-mini
```

然后把这两个文件所在目录设置给：

```bash
export ONNX_MODEL_DIR=/path/to/Step-Audio-2-mini/token2wav
```

### 4.3 中文 WER 所需 ASR 模型

`run_wer.py` 使用 FunASR 的中文 ASR 模型。推荐准备 `funasr/paraformer-zh` 到本地：

```bash
huggingface-cli download funasr/paraformer-zh \
  --local-dir /path/to/paraformer-zh
```

然后设置：

```bash
export PARAFORMER_MODEL=/path/to/paraformer-zh
```

如果你已经有一个能被 `funasr.AutoModel(model=...)` 正常加载的本地 Paraformer 目录，也可以直接指向自己的路径。

### 4.4 SIM 所需 WavLM checkpoint

说话人相似度评测使用 `wavlm_large_finetune.pth`。`seed-tts-eval` 官方 README 给出的公开链接是：

- [Google Drive: wavlm_large_finetune.pth](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)

下载后设置：

```bash
export SPEAKER_CKPT=/path/to/wavlm_large_finetune.pth
```

如果不设置这个文件，流水线仍然可以跑完生成和 WER，但会跳过 SIM。

## 5. 数据集准备

默认使用 `seed-tts-eval` 中文测试集：

```bash
export EVAL_META_PATH=/path/to/seedtts_testset_zh/zh/meta.lst
export EVAL_DATA_PATH=/path/to/seedtts_testset_zh/zh
```

`meta.lst` 典型格式：

```text
utt_id|prompt_text|prompt_wav_path|infer_text
```

其中 `prompt_wav_path` 一般是相对于 `EVAL_DATA_PATH` 的相对路径，例如 `prompt-wavs/xxx.wav`。

## 6. 运行方式

### 6.1 完整流水线

```bash
cd /path/to/Video-MME/cpp-eval/tts_seed

export CPP_BIN=/path/to/llama.cpp-omni/build/bin/llama-omni-tts-eval
export MODEL_PATH=/path/to/MiniCPM-o-4_5-gguf
export ONNX_MODEL_DIR=/path/to/Step-Audio-2-mini/token2wav
export PARAFORMER_MODEL=/path/to/paraformer-zh
export SPEAKER_CKPT=/path/to/wavlm_large_finetune.pth
export EVAL_META_PATH=/path/to/seedtts_testset_zh/zh/meta.lst
export EVAL_DATA_PATH=/path/to/seedtts_testset_zh/zh

bash run_tts_eval_cpp_zh.sh
```

常用覆盖项：

```bash
GPUS_PER_NODE=1 SEED=42 bash run_tts_eval_cpp_zh.sh
GPUS_PER_NODE=4 MODEL_PATH=/path/to/model bash run_tts_eval_cpp_zh.sh
TTS_MODEL_PATH=/path/to/tts.gguf bash run_tts_eval_cpp_zh.sh
```

### 6.2 只跑评测，不重新生成

如果 `SAVE_DIR` 里已经有生成好的 wav：

```bash
bash run_eval_only.sh /path/to/save_dir
```

## 7. 输出结果

一次完整运行后，主要产物包括：

- `eval_results/cpp-zh-*/` 或你自定义的 `SAVE_DIR`
- `logs/extract_bundle_*.log`
- `logs/cpp_gpu*_*.log`
- `logs/wer_*.log`
- `logs/sim_*.log`
- `run_cpp_eval_results.txt`

`SAVE_DIR` 下常见文件：

- `*.wav`：生成结果
- `_prompt_bundles/`：参考音频对应的缓存
- `manifest_rank*.tsv`：送给 C++ 的输入清单
- `wav_res_ref_text.wer`：WER 汇总
- `wav_res_ref_text.sim`：SIM 汇总

## 8. 常见问题

### `Unknown argument: --teacher-forcing`

说明你的 `llama-omni-tts-eval` 不是打过 `tts_seed/diff-master.patch` 的版本，重新打补丁并编译。

### `C++ binary not found`

检查 `CPP_BIN` 是否指向：

```bash
/path/to/llama.cpp-omni/build/bin/llama-omni-tts-eval
```

### `Speaker checkpoint not found`

只会跳过 SIM，不影响生成和 WER。补上 `SPEAKER_CKPT` 后可重新执行：

```bash
bash run_eval_only.sh /path/to/save_dir
```

### `No module named ...` / `torch.hub` 找不到 `s3prl-main`

请确认：

```bash
ls /path/to/Video-MME/cpp-eval/tts_seed/eval_tools/s3prl-main
```

或者设置：

```bash
export S3PRL_REPO=/path/to/s3prl
```

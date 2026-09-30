# Gemma-3-270M 需要授权

RLinf RECAP Step 2 依赖 `google/gemma-3-270m`（gated）。

当前 HF 账号 `FelishLeon` **尚未获得访问权限**（403 GatedRepoError）。

## 操作步骤
1. 浏览器登录 Hugging Face（账号需与本机 token `FelishLeon` 一致）
2. 打开 https://huggingface.co/google/gemma-3-270m
3. 同意 Google Gemma 使用协议 / 申请访问
4. 通知我，我会继续下载到：
   `/root/rlinf_recap/checkpoints/gemma-3-270m`

也可自行执行：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
hf download google/gemma-3-270m --local-dir /root/rlinf_recap/checkpoints/gemma-3-270m
```

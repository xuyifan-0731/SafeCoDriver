# Git 上传指南 (本地参考，不上传)

> 本文档说明如何将 SafeCoDriver 项目推送到 GitHub。供 AI agent 参考。

---

## 前置条件

1. **代理可用**: 本机通过 `http://127.0.0.1:18891` 代理访问 GitHub
2. **Git 配置**: `~/.gitconfig` 已配置 proxy
3. **SSH key**: 已配置 `git@github.com` SSH 访问

验证连通性:
```bash
curl -s --connect-timeout 5 -x http://127.0.0.1:18891 https://github.com -o /dev/null -w "%{http_code}"
# 应返回 200
```

## 仓库信息

- **Remote**: `git@github.com:xuyifan-0731/SafeCoDriver.git`
- **Branch**: `main`
- **项目路径**: `/raid/xuyifan/jiqiuyu/`

## 上传步骤

### 1. 进入项目目录
```bash
cd /raid/xuyifan/jiqiuyu
```

### 2. 检查状态
```bash
git status
```

### 3. 添加文件 (注意 .gitignore 排除了大文件)
```bash
# 添加所有被跟踪的修改
git add -A

# 或者选择性添加
git add coop_safety/ experiments/ docs/ paper/ README.md AGENTS.md .gitignore
```

### 4. 不要添加的内容 (.gitignore 已排除)

| 排除项 | 原因 | 大小 |
|--------|------|------|
| `data/` | DeepAccident 数据集 | ~93GB |
| `models/` | 训练好的权重 (可重新训练) | ~1MB |
| `third_party/` | V2Xverse/CARLA 等 | ~211GB |
| `experiments/results/` | 生成的实验结果 JSON | ~44MB |
| `experiments/*.log` | 运行日志 | 若干 KB |

### 5. 提交
```bash
git commit -m "描述修改内容

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### 6. 推送
```bash
git push origin main
```

## 常见问题

### 推送失败 (网络问题)
```bash
# 确认代理工作
curl -x http://127.0.0.1:18891 https://github.com

# 如果 SSH 不通，尝试 HTTPS
git remote set-url origin https://github.com/xuyifan-0731/SafeCoDriver.git
git push origin main
```

### 文件太大
```bash
# 检查哪些文件被跟踪
git ls-files --others --exclude-standard | xargs du -sh | sort -h | tail -20

# 如果意外 add 了大文件
git reset HEAD <file>
```

### 从新机器克隆后复现
```bash
git clone git@github.com:xuyifan-0731/SafeCoDriver.git
cd SafeCoDriver
conda create -n coop-safety python=3.10 -y
conda activate coop-safety
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy shapely scikit-learn

# 下载数据到 data/DeepAccident/
# 训练模型
python coop_safety/learned/train_collision.py
python coop_safety/learned/train_collision_v2.py

# 运行评测
python experiments/run_deepaccident_unified.py
```

## 已有的 git 历史

```
84a11bb Initial release: SafeCoDriver - Pluggable Safety Constraint for Cooperative Driving
```

初始提交包含 65 个文件，11574 行代码。

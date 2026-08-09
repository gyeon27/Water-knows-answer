$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimePython = "C:\Users\USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$VenvPython = Join-Path $ProjectRoot ".venv-gpu\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    & $RuntimePython -m venv (Join-Path $ProjectRoot ".venv-gpu") --system-site-packages
}
& $VenvPython -m pip install --upgrade --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
& $VenvPython -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(32).cuda().requires_grad_(); x.square().sum().backward(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

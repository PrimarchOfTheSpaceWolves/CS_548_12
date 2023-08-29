@echo off
SET PROJ_DIR=%~dp0
SET DRIVE_LETTER=%cd:~0,2%
SET BASE_DIR=%DRIVE_LETTER%\Software

SET PATH=%BASE_DIR%\PortableGit\bin;%BASE_DIR%\WPy64-310111\python-3.10.11.amd64;%BASE_DIR%\WPy64-310111\python-3.10.11.amd64\Scripts;%PATH%
SET PATH=%PATH%;./.venv/Lib/site-packages/nvidia/cuda_runtime/bin;./.venv/Lib/site-packages/nvidia/cudnn/bin;./.venv/Lib/site-packages/nvidia/cublas/bin;./.venv/Lib/site-packages/nvidia/cufft/bin;./.venv/Lib/site-packages/nvidia/curand/bin;./.venv/Lib/site-packages/nvidia/cusolver/bin;./.venv/Lib/site-packages/nvidia/cusparse/bin
START "" %BASE_DIR%\VSCode-win32-x64-1.81.1\Code.exe %PROJ_DIR%

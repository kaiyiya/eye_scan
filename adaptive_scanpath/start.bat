@echo off
REM Windows启动脚本 - AdaptiveScanPath

echo ============================================================
echo AdaptiveScanPath - 快速开始
echo ============================================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [1/4] 检查依赖...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo 警告: PyTorch未安装或导入失败
    echo 请运行: pip install -r requirements.txt
    echo.
)

echo.
echo [2/4] 运行测试...
python test_all.py
if errorlevel 1 (
    echo.
    echo 测试失败！请检查错误信息
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 测试通过！可以选择以下操作:
echo ============================================================
echo.
echo 1. 快速开始（模拟数据训练）
echo    python quickstart.py
echo.
echo 2. 完整训练（需要准备数据）
echo    python train.py
echo.
echo 3. 评估模型
echo    python eval.py --checkpoint checkpoints/best_model.pth
echo.
echo 4. 查看训练日志
echo    tensorboard --logdir=logs
echo.
pause

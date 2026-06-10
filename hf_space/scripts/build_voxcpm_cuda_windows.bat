@echo off
setlocal

set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" (
  echo Visual Studio developer command prompt was not found.
  exit /b 1
)

if "%VOXCPM_CUDA_ARCHITECTURES%"=="" set "VOXCPM_CUDA_ARCHITECTURES=89"

call "%VSDEVCMD%" -arch=x64
if errorlevel 1 exit /b %errorlevel%

cmake -G "Visual Studio 17 2022" -A x64 -S hf_space\VoxCPM.cpp -B hf_space\VoxCPM.cpp\build-cuda -DVOXCPM_BUILD_TESTS=OFF -DVOXCPM_BUILD_BENCHMARK=OFF -DVOXCPM_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=%VOXCPM_CUDA_ARCHITECTURES%
if errorlevel 1 exit /b %errorlevel%

cmake --build hf_space\VoxCPM.cpp\build-cuda --config Release --target voxcpm-server
if errorlevel 1 exit /b %errorlevel%

echo Built hf_space\VoxCPM.cpp\build-cuda\examples\Release\voxcpm-server.exe

@echo off
setlocal

set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" (
  echo Visual Studio developer command prompt was not found.
  exit /b 1
)

call "%VSDEVCMD%" -arch=x64
if errorlevel 1 exit /b %errorlevel%

cmake -G "Visual Studio 17 2022" -A x64 -S hf_space\VoxCPM.cpp -B hf_space\VoxCPM.cpp\build -DVOXCPM_BUILD_TESTS=OFF -DVOXCPM_BUILD_BENCHMARK=OFF -DVOXCPM_CUDA=OFF
if errorlevel 1 exit /b %errorlevel%

cmake --build hf_space\VoxCPM.cpp\build --config Release --target voxcpm_tts voxcpm-server
if errorlevel 1 exit /b %errorlevel%

echo Built hf_space\VoxCPM.cpp\build\examples\Release\voxcpm_tts.exe
echo Built hf_space\VoxCPM.cpp\build\examples\Release\voxcpm-server.exe

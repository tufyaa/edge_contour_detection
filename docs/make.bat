@ECHO OFF

set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "clean" (
    rmdir /s /q %BUILDDIR%
    rmdir /s /q autoapi
    goto end
)

if "%1" == "html" (
    %SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%/html
    goto end
)

echo Usage: make.bat [clean^|html]

:end

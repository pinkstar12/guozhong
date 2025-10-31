# Windows Conda Troubleshooting

When running PowerShell commands such as:

```powershell
(& D:\anaconda\shell\condabin\conda-hook.ps1) ; (conda activate torch)
```

PowerShell can throw the following error:

```
无法加载文件 D:\anaconda\shell\condabin\conda-hook.ps1，因为在此系统上禁止运行脚本。
CategoryInfo          : SecurityError: (:) [], PSSecurityException
FullyQualifiedErrorId : UnauthorizedAccess
```

This is caused by the PowerShell **execution policy** blocking unsigned scripts.

## Resolution

1. **Start PowerShell as Administrator.**
2. Check the current execution policy:
   ```powershell
   Get-ExecutionPolicy -List
   ```
3. Allow locally created scripts to run by setting the current user's policy to `RemoteSigned`:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```
   If you only want the change for the current PowerShell session, use:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
4. Retry the Conda activation command.

## Ensure `conda` is on PATH

If you still see `conda : 无法将“conda”项识别为 ...` it means the `conda` executable is not on your PATH.

- Run `where conda` to locate the Conda installation.
- Add `D:\anaconda\Scripts` and `D:\anaconda` to your PATH environment variable.
- Alternatively, call Conda using its full path:
  ```powershell
  & "D:\anaconda\Scripts\conda.exe" run --live-stream --name torch python d:/project/guozhong/airbattle/demo.py
  ```

After updating PATH or using the explicit path, open a new PowerShell window and run the activation or `conda run` command again.

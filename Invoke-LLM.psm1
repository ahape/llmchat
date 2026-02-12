<#
.SYNOPSIS
  Wrapper for chat.py - Query LLMs via OpenAI-compatible API routers

.EXAMPLES
  Invoke-LLM "why is the sky blue?"
  Invoke-LLM "explain quantum computing" -Model "deepseek-ai/DeepSeek-V3.2"
  Invoke-LLM -ListModels
  Invoke-LLM -SwitchModel
  Invoke-LLM -SwitchRouter
  Invoke-LLM "continue our conversation" -Context
  Invoke-LLM "start fresh conversation"
  Invoke-LLM -Compose
  Invoke-LLM -Compose -Model "deepseek-ai/DeepSeek-V3.2"

  Aliases:
  Ask-LLM "why is the sky blue?"
#>
function Invoke-LLM {
  [CmdletBinding()]
  param(
    [Parameter(Position=0, ValueFromPipeline=$true, HelpMessage="The question to ask the LLM")]
    [string]$Question,

    [Parameter(HelpMessage="Model name")]
    [string]$Model,

    [Parameter(HelpMessage="List available models")]
    [switch]$ListModels,

    [Parameter(HelpMessage="Interactively switch the default model")]
    [switch]$SwitchModel,

    [Parameter(HelpMessage="Interactively switch the API router")]
    [switch]$SwitchRouter,

    [Parameter(HelpMessage="Open Neovim to compose your question")]
    [switch]$Compose,

    [Parameter(HelpMessage="Continue conversation from last message (without this flag, starts fresh)")]
    [switch]$Context
  )

  $sourceRoot = "C:\Users\AlanHape\source\repos\llmchat\" # Change this when copying to other directories
  $pythonExe = Join-Path $sourceRoot ".venv/Scripts/python.exe"
  $pythonArgs = @(Join-Path $sourceRoot 'chat.py')

  if ($Question) {
    $pythonArgs += $Question
  }
  if ($Model) {
    $pythonArgs += @("--model", $Model)
  }
  if ($ListModels) {
    $pythonArgs += "--list-models"
  }
  if ($SwitchModel) {
    $pythonArgs += "--switch-model"
  }
  if ($SwitchRouter) {
    $pythonArgs += "--switch-router"
  }
  if ($Compose) {
    $pythonArgs += "--compose"
  }
  if ($Context) {
    $pythonArgs += "--context"
  }

  try {
      Push-Location $sourceRoot
      & $pythonExe @pythonArgs
  } finally {
      Pop-Location
  }
}

# Export the function and create alias
New-Alias -Name Ask-LLM -Value Invoke-LLM

Export-ModuleMember -Function Invoke-LLM
Export-ModuleMember -Alias Ask-LLM

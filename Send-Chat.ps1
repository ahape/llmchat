<#
.SYNOPSIS
  Wrapper for chat.py - Query LLMs via OpenAI-compatible API routers

.EXAMPLES
  Send-Chat "why is the sky blue?"
  Send-Chat "explain quantum computing" -Model "deepseek-ai/DeepSeek-V3.2"
  Send-Chat -ListModels
  Send-Chat -SwitchModel
  Send-Chat -SwitchRouter
  Send-Chat "continue our conversation" -Context
  Send-Chat "start fresh" -Context new
  Send-Chat -Compose
  Send-Chat -Compose -Model "deepseek-ai/DeepSeek-V3.2"
#>
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

  [Parameter(HelpMessage="Context ID (use 'new' for fresh context, or omit value for default)")]
  [string]$Context
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
  $pythonArgs += @("--context", $Context)
} elseif ($PSBoundParameters.ContainsKey('Context') -and [string]::IsNullOrEmpty($Context)) {
  # Context switch was used without a value
  $pythonArgs += "--context"
}

try {
    Push-Location $sourceRoot
    & $pythonExe @pythonArgs
} finally {
    Pop-Location
}

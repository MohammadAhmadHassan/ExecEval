<#
Extract CodeHalu tasks rejected by the validation sweep into separate JSON files.

Outputs, next to this script:
  no_solution_tasks.json  - tasks whose manifest verdict is NO_SOLUTION
  unusable_tasks.json     - tasks whose manifest verdict is UNUSABLE

Each output entry keeps the task statement and metadata once, and groups all of
the task's original test cases under `unittests`.
#>

$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$manifestPath = Join-Path $root 'codehalu_manifest.csv'

$sourceFiles = @(
    'calculate_boundary_hallucination.json',
    'data_compliance_hallucination.json',
    'external_source_hallucination.json',
    'identification_hallucination.json',
    'logic_breakdown.json',
    'logic_deviation.json',
    'physical_constraint_hallucination.json',
    'structural_access_hallucination.json'
)

$mainCategory = @{
    'data_compliance_hallucination.json'     = 'Mapping'
    'structural_access_hallucination.json'   = 'Mapping'
    'identification_hallucination.json'      = 'Naming'
    'external_source_hallucination.json'     = 'Naming'
    'physical_constraint_hallucination.json' = 'Resource'
    'calculate_boundary_hallucination.json'  = 'Resource'
    'logic_deviation.json'                   = 'Logic'
    'logic_breakdown.json'                   = 'Logic'
}

if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Manifest not found: $manifestPath"
}

$manifestRows = Import-Csv -LiteralPath $manifestPath
$verdictByTask = @{}
foreach ($entry in $manifestRows) {
    if ($entry.verdict -in @('NO_SOLUTION', 'UNUSABLE')) {
        $key = '{0}|{1}' -f $entry.source_file, $entry.task_id
        $verdictByTask[$key] = $entry
    }
}

$noSolutionTasks = New-Object 'System.Collections.Generic.List[object]'
$unusableTasks = New-Object 'System.Collections.Generic.List[object]'

foreach ($sourceFile in $sourceFiles) {
    $sourcePath = Join-Path $root $sourceFile
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Source JSON not found: $sourcePath"
    }

    Write-Host "Reading $sourceFile"
    $rows = Get-Content -LiteralPath $sourcePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskById = @{}

    foreach ($row in $rows) {
        $key = '{0}|{1}' -f $sourceFile, $row.task_id
        if (-not $verdictByTask.ContainsKey($key)) { continue }

        $manifestEntry = $verdictByTask[$key]
        $verdict = $manifestEntry.verdict
        if (-not $taskById.ContainsKey([string]$row.task_id)) {
            $solutions = @()
            if (-not [string]::IsNullOrWhiteSpace([string]$row.solutions)) {
                try {
                    $parsedSolutions = ConvertFrom-Json -InputObject ([string]$row.solutions)
                    if ($parsedSolutions -is [array]) {
                        $solutions = $parsedSolutions
                    } elseif ($null -ne $parsedSolutions) {
                        $solutions = @($parsedSolutions)
                    }
                } catch {
                    $solutions = @([string]$row.solutions)
                }
            }

            $task = [PSCustomObject][ordered]@{
                task_id             = $row.task_id
                source_file         = $sourceFile
                halu_type           = $row.halu_type
                main_category       = $mainCategory[$sourceFile]
                verdict             = $verdict
                fail_outcome        = $manifestEntry.fail_outcome
                n_solutions_tried   = if ([string]::IsNullOrWhiteSpace($manifestEntry.n_solutions_tried)) { 0 } else { [int]$manifestEntry.n_solutions_tried }
                n_testcases         = [int]$manifestEntry.n_testcases
                difficulty          = $row.difficulty
                question            = $row.question
                starter_code        = $row.starter_code
                url                 = $row.url
                solutions           = $solutions
                unittests           = (New-Object 'System.Collections.Generic.List[object]')
            }
            $taskById[[string]$row.task_id] = $task
            if ($verdict -eq 'NO_SOLUTION') {
                $noSolutionTasks.Add($task)
            } else {
                $unusableTasks.Add($task)
            }
        }

        $taskById[[string]$row.task_id].unittests.Add([PSCustomObject][ordered]@{
            id            = $row.id
            test_case_id  = $row.test_case_id
            input         = $row.input
            output        = $row.output
        })
    }

    Remove-Variable rows, taskById
}

$noSolutionPath = Join-Path $root 'no_solution_tasks.json'
$unusablePath = Join-Path $root 'unusable_tasks.json'
ConvertTo-Json -InputObject $noSolutionTasks.ToArray() -Depth 100 |
    Set-Content -LiteralPath $noSolutionPath -Encoding UTF8
ConvertTo-Json -InputObject $unusableTasks.ToArray() -Depth 100 |
    Set-Content -LiteralPath $unusablePath -Encoding UTF8

Write-Host "NO_SOLUTION tasks: $($noSolutionTasks.Count) -> $noSolutionPath"
Write-Host "UNUSABLE tasks:    $($unusableTasks.Count) -> $unusablePath"

param(
    [Parameter(Mandatory=$true)][string]$EndpointId,
    [Parameter(Mandatory=$true)][string]$ReferenceAudio,
    [Parameter(Mandatory=$true)][string]$ReferenceTextFile,
    [string]$TextFile = "$PSScriptRoot/../samples/thai-700.txt",
    [ValidateRange(1,5)][int]$Runs = 1,
    [ValidateRange(30,300)][int]$TimeoutSeconds = 120,
    [string]$OutputDirectory = "$PSScriptRoot/../benchmark-results"
)
$ErrorActionPreference = 'Stop'
$headers = @{}
try {
    if ($EndpointId -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid EndpointId' }
    $key = $env:RUNPOD_API_KEY
    if (-not $key) {
        $secure = Read-Host 'RunPod API key (hidden)' -AsSecureString
        $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        try { $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
    }
    $headers = @{ Authorization = "Bearer $key" }
    $base = "https://api.runpod.ai/v2/$EndpointId"
    $text = [IO.File]::ReadAllText((Resolve-Path $TextFile).Path, [Text.Encoding]::UTF8).Trim()
    $transcript = [IO.File]::ReadAllText((Resolve-Path $ReferenceTextFile).Path, [Text.Encoding]::UTF8).Trim()
    $audio = [IO.File]::ReadAllBytes((Resolve-Path $ReferenceAudio).Path)
    if ($audio.Length -gt (12 * 1024 * 1024)) { throw 'Reference audio exceeds 12 MB' }
    if ($text.Length -eq 0 -or $text.Length -gt 2000) { throw 'Text must contain 1-2000 characters' }
    if ($transcript.Length -eq 0) { throw 'Reference transcript is required' }
    $payload = @{ input = @{ action = 'generate'; text = $text; reference_text = $transcript;
        reference_audio_b64 = [Convert]::ToBase64String($audio); inference_timesteps = 10;
        cfg_value = 2.0; seed = 42; modifier = @{ speed = 1.0; pitch_semitones = 0; volume = 1.0 } };
        policy = @{ executionTimeout = $TimeoutSeconds * 1000; ttl = $TimeoutSeconds * 1000 } }
    $body = [Text.Encoding]::UTF8.GetBytes(($payload | ConvertTo-Json -Depth 8 -Compress))
    if ($body.Length -gt 19000000) { throw 'Input JSON exceeds the benchmark payload limit' }
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $csvPath = Join-Path $OutputDirectory 'results.csv'
    for ($n = 1; $n -le $Runs; $n++) {
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $jobId = $null
        try {
            # /runsync permits 20 MB payloads; return after 1 sec then poll normally.
            $submitted = Invoke-RestMethod -Method Post -Uri "$base/runsync?wait=1000" -Headers $headers -ContentType 'application/json; charset=utf-8' -Body $body -TimeoutSec 20
            $jobId = $submitted.id
            if (-not $jobId) { throw 'RunPod did not return a job ID' }
            $status = $submitted
            while ($status.status -in @('IN_QUEUE','IN_PROGRESS')) {
                if ($timer.Elapsed.TotalSeconds -gt $TimeoutSeconds) { throw 'Benchmark timeout; job cancellation requested' }
                Start-Sleep -Milliseconds 1000
                $status = Invoke-RestMethod -Method Get -Uri "$base/status/$jobId" -Headers $headers -TimeoutSec 20
            }
            if ($status.status -ne 'COMPLETED') { throw "RunPod status: $($status.status)" }
            $timer.Stop()
            $out = $status.output
            if ($out.backend -ne 'nano-vllm' -or -not $out.audio_b64) { throw 'Wrong image/backend or missing WAV output' }
            $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
            [IO.File]::WriteAllBytes((Join-Path $OutputDirectory "$stamp.wav"), [Convert]::FromBase64String($out.audio_b64))
            $row = [pscustomobject]@{
                time_utc = [DateTime]::UtcNow.ToString('o'); job_id = $jobId; worker_id = $out.worker_id;
                request_index = $out.request_index; characters = $out.characters;
                end_to_end_seconds = [Math]::Round($timer.Elapsed.TotalSeconds,3);
                delay_seconds = $status.delayTime / 1000; execution_seconds = $status.executionTime / 1000;
                model_load_seconds = $out.model_load_seconds; reference_encode_seconds = $out.reference_encode_seconds;
                generation_seconds = $out.generation_seconds; duration_seconds = $out.duration_seconds;
                rtf = $out.rtf; segments = $out.segments; reference_cache_hit = $out.reference_cache_hit;
                under_60_seconds = ($timer.Elapsed.TotalSeconds -lt 60); model_source = $out.model_source;
                status = 'completed'
            }
            $row | Export-Csv -Path $csvPath -Append -NoTypeInformation -Encoding UTF8
            $row | Format-List
        } catch {
            if ($jobId) {
                try { Invoke-RestMethod -Method Post -Uri "$base/cancel/$jobId" -Headers $headers -TimeoutSec 10 | Out-Null } catch { }
            }
            # Never print input, audio/base64, key, or signed bridge URLs.
            throw ('Benchmark failed: ' + $_.Exception.Message)
        }
    }
} finally {
    $key = $null
    $headers.Clear()
    if ($secure) { $secure.Dispose() }
}

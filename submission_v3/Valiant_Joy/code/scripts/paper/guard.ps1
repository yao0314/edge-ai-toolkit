# Safety guard for long measurement blocks. Samples every 20 s and prints a line ONLY when a
# threshold trips, so the monitor stays quiet unless something is wrong.
#   CPU      : whole-machine %, from PerfOS_Processor _Total (LoadPercentage is unreliable here)
#   RAM      : free physical memory
#   Throttle : "% Processor Performance" of _Total. The temperature sensor needs administrator
#              rights on this laptop, so a sustained clock drop is the throttling proxy.
param(
  [int]$IntervalSec = 20,
  [int]$CpuTrip = 92,        # sustained whole-machine CPU %
  [double]$RamFreeGbTrip = 1.2,
  [int]$PerfTrip = 70,       # sustained % of nominal clock
  [int]$Consecutive = 3      # how many samples in a row before it counts
)
$cpuHits = 0; $perfHits = 0; $lastReport = [datetime]::MinValue
while ($true) {
  try {
    $cpu = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
    $os = Get-CimInstance Win32_OperatingSystem
    $ramFree = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $perf = (Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'").PercentProcessorPerformance
  } catch { Start-Sleep -Seconds $IntervalSec; continue }

  if ($cpu -ge $CpuTrip) { $cpuHits++ } else { $cpuHits = 0 }
  # A low clock only means throttling when the CPU is actually being asked to work: Windows
  # downclocks an idle machine to well under 50% of nominal, which is not a fault condition.
  if ($perf -gt 0 -and $perf -le $PerfTrip -and $cpu -ge 50) { $perfHits++ } else { $perfHits = 0 }

  $msgs = @()
  if ($cpuHits -ge $Consecutive) { $msgs += "CPU pegged at $cpu% for $($cpuHits * $IntervalSec)s" }
  if ($ramFree -le $RamFreeGbTrip) { $msgs += "free RAM down to $ramFree GB" }
  if ($perfHits -ge $Consecutive) { $msgs += "clock down to $perf% of nominal for $($perfHits * $IntervalSec)s (throttling proxy)" }

  if ($msgs.Count -gt 0 -and ((Get-Date) - $lastReport).TotalSeconds -ge 120) {
    $top = (Get-Process | Sort-Object CPU -Descending | Select-Object -First 3 |
            ForEach-Object { "$($_.Name)" }) -join ', '
    "GUARD: " + ($msgs -join '; ') + " | busiest: $top"
    [Console]::Out.Flush()
    $lastReport = Get-Date
  }
  Start-Sleep -Seconds $IntervalSec
}

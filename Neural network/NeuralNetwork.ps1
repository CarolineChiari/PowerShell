Using module ".\Neuron.psm1"

class NeuralNetwork {
    [Neuron[][]]$Network  
    
    NeuralNetwork (
        [int[]]$NetworkLayout, # Each array index contains the number of neurons in the layer
        [float]$LearningRate
    ) {
        $this.Network = @()
        for ($i = 0; $i -lt $NetworkLayout.Count; $i++) {
            $layer = @()
            for ($n = 0; $n -lt $NetworkLayout[$i]; $n++) {
                if ($i -eq 0) {
                    $layer += [Neuron]::new($null, $NetworkLayout[0], $learningRate)
                }
                else {
                    $layer += [Neuron]::new($this.Network[$i - 1], $NetworkLayout[$i - 1], $learningRate)
                }
                
            }
            $this.Network += , $layer
            # write-host "Layer: $i" -ForegroundColor Green
            # write-host $this.network.count
            # foreach ($neuron in $layer){
            #     $neuron.print()
            # }
            
        }
    }

    [float[]]GetOutput([float[]]$InputValues) {
        $output = @()
        for ($i = 0; $i -lt $this.Network.count; $i++) {
            if ($i -eq 0) {
                foreach ($neuron in $this.Network[$i]) {
                    # $neuron | gm | write-host
                    $neuron.GetOutput($InputValues) # | out-null
                }
            }
            else {
                foreach ($neuron in $this.Network[$i]) {
                    $neuron.GetOutput() | out-null
                }
            }
            if ($i -eq $this.Network.count - 1) {
                foreach ($neuron in $this.Network[$i]) {
                    $output += $neuron.Value
                }
            }
        }
        return $output
    }

    [void]backPropagate([float[]]$expectedValues, [float[]]$InputValues) {
        for ($i = $this.Network.count - 1; $i -ge 0; $i--) {
            for ($n = 0 ; $n -lt $this.Network[$i].count; $n++) {
                if ($i -eq $this.Network.count - 1) {
                    $this.Network[$i][$n].backPropagate($expectedValues[$n])
                }
                elseif ($i -eq 0) {
                    $this.Network[$i][$n].backPropagate($InputValues)
                }
                else {
                    $this.Network[$i][$n].backPropagate()
                }
            }
        }
    }
    print() {
        Write-Host "----------------" -foregroundcolor red
        for ($i = 0; $i -lt $this.Network.count; $i++) {
            write-host "Layer: $i" -ForegroundColor Green
            foreach ($neuron in $this.Network[$i]) {
                $neuron.print()
            }
        }
    }
    print([int]$x,[int]$n){
        $this.Network[$x][$n].print()
    }
}

Clear-Host

$set = @()
for ($i = 0 ; $i -le 100; $i++) {
    for ($j = 0 ; $j -le 100; $j++) {
        $set += , @(($i/10), ($j/10), $($i -gt $j ? 1 : 0))
    }
}
# write-host $set[0].Count
# $set = $set | sort-object { Get-Random }
# write-host $set[0].Count
# return
# $set[100]
# return
for ($lr = 1; $lr -lt 2; $lr++) {
    $lr = 0.01
    $network = [NeuralNetwork]::new(@(2, 2, 2, 1), $lr / 100)
    # $network.print()
    for ($epoch = 0 ; $epoch -lt 100; $epoch++) {
        $set = $set | sort-object { Get-Random }
        $sum_error = 0
        for ($i = 0 ; $i -lt $set.count; $i++) {
            $inputValues = @($set[$i][0], $set[$i][1])
            $output = $network.GetOutput($inputValues)
            $network.backPropagate($set[$i][2], $inputValues)
            # write-host $input -ForegroundColor Green
            # write-host $output -ForegroundColor Green
            $sum_error += $set[$i][2] - $output[0]
            # $network.print(0, 0)
        }
        write-host "$epoch`: $sum_error" -ForegroundColor Green
    }
    # $network.print()
    Write-Host "Learning Rate: $($lr/100)" -ForegroundColor Green
    for ($i = 0; $i -le 10; $i++) {
        for ($j = 0; $j -le 10; $j++) {
            $out = $network.GetOutput(@($i, $j))
            $color = "Red"
            if (($out -gt .5 -and $i -gt $j) -or ($out -lt .5 -and $i -le $j)) {
                $color = "Green"
            }
            write-host "$i,$j --> $($network.GetOutput(@($i, $j))) | $($i -gt $j ? 1 : 0)" -foregroundcolor $color
        }
    }
}

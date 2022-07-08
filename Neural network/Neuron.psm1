
class Neuron {
    [int32] hidden $NumberOfInputs
    [float[]] hidden $Weights
    [Neuron[]] hidden $InputNeurons
    [bool] hidden $IsInput
    [float[]] hidden $NetworkErrors
    [float] hidden $LearningRate
    
    [float]$Value

    Neuron (
        [Neuron[]]$InputNeurons,
        [int]$NumberOfInputs,
        [float]$LearningRate
    ) {
        # Contructor
        # If no Input Neurons are given, assume it's an input neuron
        $this.LearningRate = $LearningRate
        $this.NumberOfInputs = $numberOfInputs
        # Write-Host ($InputNeurons.Count -eq 0)
        if ($InputNeurons.count -eq 0) {
            # Write-Host ($InputNeurons.Count -eq 0)
            $this.IsInput = $true
        }
        else {
            $this.InputNeurons = $InputNeurons
            $this.IsInput = $false
        }
        # write-host $this.IsInput
        # Initialize the weights
        $this.Weights = [float[]]::new($NumberOfInputs + 1) # +1 for the bias
        for ($i = 0; $i -lt $numberOfInputs + 1; $i++) { 
            $this.Weights[$i] = (Get-Random -Min -10 -Max 10) / 10 # Weight is random number between -1 and 1
        }

        $this.NetworkErrors = @()
        $this.value = 0
    }

    # Calculate the output of the neuron (for an input Neuron)
    [float] GetOutput([float[]]$InputValues) {
        $exp = [float]0
        for ($i = 0; $i -lt $this.NumberOfInputs; $i++) {
            $exp += $this.Weights[$i] * $inputValues[$i]
        }
        $exp += $this.Weights[$this.NumberOfInputs] # Add the bias

        $output = [float](1.0 / (1.0 + [System.Math]::Exp(-1*$exp))) # Sigmoid function
        $this.value = $output
        return $output
    }

    [float] GetOutput() {
        $exp = [float]0
        for ($i = 0; $i -lt $this.NumberOfInputs; $i++) {
            $exp += $this.Weights[$i] * $this.InputNeurons[$i].value
        }
        $exp += $this.Weights[$this.NumberOfInputs] # Add the bias

        $output = [float](1.0 / (1.0 + [System.Math]::Exp(-1*$exp))) # Sigmoid function
        $this.value = $output
        return $output
    }

    [float]GetDerivative() {
        return $this.value * (1.0 - $this.value)
    }

    [float]GetError([float] $target) {
        # Calculate the error of the neuron for an output layer neuron
        return ($this.value - $target ) * $this.GetDerivative()
    }

    [float]GetError() {
        # For non-output neurons
        $err = 0
        foreach ($ne in $this.NetworkErrors) {
            $err += $ne
        }
        # write-host "--"
        # write-host $err
        # write-host $this.NetworkErrors
        $this.NetworkErrors = @()
        
        return $err * $this.GetDerivative()
    }

    AddError([float] $Err) {
        $this.NetworkErrors += $Err
    }

    backPropagate([float] $target) {
        #Output layer
        $err = $this.GetError($target)
        # write-host "$($this.value) - $target = $err"
        for ($i = 0; $i -lt $this.InputNeurons.count; $i++) {
            $this.InputNeurons[$i].AddError($err * $this.Weights[$i])
            $this.Weights[$i] += -1 * $this.InputNeurons[$i].value * $err * $this.LearningRate
        }
        $this.Weights[$this.InputNeurons.count] += -1 * $err * $this.LearningRate
    }

    backPropagate() {
        $err = $this.GetError()
        for ($i = 0; $i -lt $this.InputNeurons.count; $i++) {
            $this.InputNeurons[$i].AddError($err * $this.Weights[$i])
            $this.Weights[$i] += -1 * $this.InputNeurons[$i].value * $err * $this.LearningRate
        }
        $this.Weights[$this.InputNeurons.count] += -1 * $err * $this.LearningRate
    }

    backPropagate([float[]]$NetworkInputs) {
        #Input layer
        $err = $this.GetError()
        # return
        for ($i = 0; $i -lt $NetworkInputs.count; $i++) {
            $this.Weights[$i] += -1 * $NetworkInputs[$i] * $err * $this.LearningRate
        }
        $this.Weights[$NetworkInputs.count] += -1 * $err * $this.LearningRate
    }

    print() {
        Write-host "========"
        write-host "`tI am $($this.IsInput? '':'NOT ')an input neuron"
        write-host "`tI have $($this.NumberOfInputs) inputs"
        write-host "`tI have $($this.InputNeurons.count) input neurons"
        write-host "`tMy weights are: "
        for ($i = 0; $i -lt $this.Weights.count; $i++) {
            write-host "`t`t$($this.Weights[$i])"
        }
        
        write-host "`tMy value is: $($this.Value)"
        write-host "========"
    }
}
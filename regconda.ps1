# origin : https://github.com/conda/conda/issues/626#issuecomment-277457230
$CondaPath = "$env:C:\Anaconda3"

function Register-Conda {
    Param (
        [Parameter(Mandatory=$True)][string]$EnvName
    )
    $envPath = "$CondaPath\envs\$EnvName"

    # error chk
    #If (!(exists -d $envPath)) {
    #  throw "Conda env '$envName' does not exist!"
    #}
    $env:Path = $($env:Path -Split ";" | ? { !($_.StartsWith($CondaPath)) }) -Join ";"
    $env:Path = "$envPath\Scripts" + ";" + $env:Path
    $env:Path = $envPath + ";" + $env:Path
    $env:CONDA_DEFAULT_ENV = $EnvName
    $env:CONDA_PREFIX = $envPath
}; Set-Alias regconda Register-Conda

# it's my env
regconda tf

# check
pip -V

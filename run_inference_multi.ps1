$algorithms = @(
    # "generative-model"
    # ,
    # "mpd"
    # ,
    # "mpd-splines"
    #,
    # "rrt"
    # ,
    # "rrt-smooth"
    # ,
    # "gpmp2"
    # ,
    "grad"
    ,
    "grad-splines"
    ,
    # "rrt-gpmp2"
    # ,
    "rrt-grad"
    ,
    "rrt-grad-splines"
)

foreach ($algo in $algorithms) {
    Write-Host "Running inference for $algo..."
    python scripts/inference.py --algorithm $algo
}

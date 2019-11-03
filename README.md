# DeepMD Active Learning Generator

## Description

This is the essential feature of the early stage [DeepMD](https://github.com/deepmodeling/deepmd-kit), and it has been merged to the DeepMD now.

Basically we can train the network adaptively according to the current feedback. Namely, the scheme generates configurations that can't be predicted well and does DFT single point calculation on them. Therefore, it makes DeepMD more feasible.

## How to use

### Get POSCAR

Firstly, you can get a desired initial configuration from material project in `POSCAR` format  in the main directory of the package.

### Produce initial random configurations

`perp.py` will produce initial random files 

```
python perp.py -h
usage: perp.py [-h] [-r] [-out OUTPUT] [-ra RATIO] [-div DIVIDES] [-si SIGMA]
               [-den DENSITY]
               [files [files ...]]

get perturbation from the exist POSCAR

positional arguments:
  files                 list of POSCAR files (plain)

optional arguments:
  -h, --help            show this help message and exit
  -r, --recursive       scan recursively for OUTCAR files
  -out OUTPUT, --output OUTPUT
                        Set the output iter parent directory of the POSCAR
  -ra RATIO, --ratio RATIO
                        Set the configuration change ratio
  -div DIVIDES, --divides DIVIDES
                        Set the divides of the density
  -si SIGMA, --sigma SIGMA
                        Set the std_devi of the normal distribution
  -den DENSITY, --density DENSITY
                        Set the center density of the whole phase graph
```

### Start active learning

Specify the initial iteration path, which can be obtained from `perp.py`

```
"sys_dir": "/home/tgzhou/Research/DeepMD/Hydrogen/perp_data/iter"
```

Then input other parameters in the `my_param.json`

The entry of the package is `param.py`, which reads the parameter from json file, for instance, `my_param.json`.

```
python param.py my_param.json -c &> run.log
```

You can use `python3 param.py -h` to check the optional argument.

```
python param.py -h
usage: param.py [-h] [-c] file

Deepmd ab-inito genrator

positional arguments:
  file                  Get the json configuration file

optional arguments:
  -h, --help            show this help message and exit
  -c, --continue_train  continue from the generator_checkpoint.json
```

``--continue_train`` argument will restart the active training scheme.

After calculation, you will see the output directory.

```
iter iter_0 iter_1 ... iter_n
```

Each `iter` directory has three subdirectories

```
deepmd  lammps  vasp
```

You can find a tensorflow graph model in `deepmd` directory

```
ls deepmd/graph_0

checkpoint  deepmd.json  frozen_model.pb  lcurve.out  model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta  stat.avg.out  stat.std.out
```


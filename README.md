## Installation
```
$ git clone https://github.com/ekinugurel/GPSImpute.git
$ cd ./GPSImpute
$ pip install -e .
```


## About
GPSImpute is a library to impute missing values for mobile datasets, largely in the context of human mobility. It uses multi-task Gaussian processes to infer missing GPS traces. In practice, for short gaps in continuous coverage, our method is comparable to existing smoothing techniques. For long gaps, however, our method cleverly takes advantage of an individual's longitudinal data to predict missing segments.

## Usage
See examples.

## Licensing

See the [LICENSE](LICENSE) file for licensing information as it pertains to files in this repository.

## Contact
Ekin Uğurel (ugurel [at] uw.edu)

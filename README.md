# Shallow Fake

Low quality video manipulation using the high quality [menpo project](https://www.menpo.org).

## Setup

Tested on os x 10.13

- Install avconv (e.g. with `brew`)
- Install python environment using [anaconda](https://anaconda.org):
    ```bash
    conda env create -n <your_env_name> -f shallow_fake.yml
    ```
- Install python-opencv (check `import cv2` works)
# WiFall

Wireless perception fall detection system, as a collection terminal project for [WiGuard](https://github.com/saurlax/wiguard).

## How to run

First, download and install the dependencies.

```bash
$ git clone https://github.com/saurlax/wifall.git
$ cd wifall
$ pip install .
```

Afterwards, you need to provide config to make the program work correctly. You can rename the `config.example.toml` to `config.toml` and make changes based on it.

Finally, run the program.

```bash
$ wifall
```

If you want to test or further develop, you can use the following command to install so that the modifications can take effect:

```bash
$ pip install --editable .
```
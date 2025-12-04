# quest-demo
QUEST-DEMO

## Conda Environment
```
conda create -n  quest  python=3.10.16
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r  requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download  en_core_web_md
```

## Vector Database Installation

The secondary index of quest needs to be built on a vector database.
You need to install PostgreSQL along with its vector extension, **pgvector**.
Alternatively, you can use Huawei’s **openGauss**.
It is recommended to use a Docker container for installation.

After installation, connect to the database using the Python SDK.
Replace the following configuration with your own database settings, and fill it in at
`quest/db/connector/connector.py`, in the function `create_opengauss_engine` at line 62:

```
    # Database connection parameters
    db_config = {
        "host": "127.0.0.1",
        "port": 18874, # Port mapped to host during database deployment
        "database": "testdb",
        "user": "remote_user",
        "password": "dmE43-3654"
    }
```

For detailed installation instructions, refer to:
openGauss:
[https://opengauss.org/en/](https://opengauss.org/en/)
PostgreSQL:
[https://www.postgresql.org/](https://www.postgresql.org/)

### Example Database Installation and Configuration

The following example demonstrates how to install and configure the openGauss database. The deployment process for PostgreSQL is similar.
**Note:** The openGauss 7.0.0-RC1 version already includes vector search functionality in its kernel, so there is no need to install an additional vector extension.
If you choose PostgreSQL, just install the pgvector plugin afterward; the syntax for both is compatible.

```
Download the image
wget https://download-opengauss.osinfra.cn/archive_test/7.0.0-RC1/openGauss7.0.0-RC1.B023/openEuler20.03/x86/openGauss-Docker-7.0.0-RC1-x86_64.tar

Load the image
sudo docker load -i openGauss-Docker-7.0.0-RC1-x86_64.tar

Start the openGauss database container
sudo docker run --name opengauss --privileged=true -d -e GS_PASSWORD=your_db_admin_password -p host_port:5432 opengauss/opengauss-server:latest -v host_persistent_storage_path:/var/lib/opengauss
```

After successfully starting the container, enter the container to create a database for testing:

First, use the admin account to create a regular user (admin accounts cannot be used for remote connections):

```
sudo docker exec -it opengauss bash
su omm
gsql -d postgres -p 5432
CREATE USER remote_user WITH PASSWORD 'your_remote_connection_password';
ALTER USER remote_user CREATEDB;
GRANT ALL PRIVILEGES TO remote_user;
```

Press `ctrl+d` to exit the admin account, then log in as the regular user to create a database for remote connections:

```
gsql -U remote_user -d postgres -p 5432
# Enter 'your_remote_connection_password'
CREATE DATABASE testdb OWNER remote_user;
```

Once the above commands are executed, the setup is complete and ready for use.


## How to Start

You can refer to `/quest/test/sf1.py` and `/quest/test/sfw1.py` for testing, as they correspond to the `SELECT FROM` statement and the `SELECT FROM WHERE` statement, respectively.


# 1. Download Helm 
```sh
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
```
```sh
chmod +x get_hem.sh
```
```sh
./get_helm.sh
```

# 2. Download the Datree binary
Go on the datree [GitHub page](https://github.com/datreeio/datree/releases/) and download the correct binary for your system.


# 3. Download Checkov
```sh
pip3 install checkov
```

# 4. Download the kubeconform binary
```sh
curl -L https://github.com/yannh/kubeconform/releases/latest/download/kubeconform-linux-amd64.tar.gz | tar xvzf -
```

# 4. Install necessary python modules
```sh
pip install -r requirements.txt
```

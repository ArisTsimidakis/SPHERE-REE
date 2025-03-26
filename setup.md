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

# 2. Download Datree
```sh
helm plugin install https://github.com/datreeio/helm-datree
```

# 3. Download Checkov
```sh
pip install checkov
```

# 4. Install necessary python modules
```sh
pip install -r requirements.txt
```

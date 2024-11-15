docker build --build-arg="USERID=$(id -u)" \
    --build-arg="GROUPID=$(id -g)" \
    --build-arg="USERNAME=$USER" \
    --build-arg="REPO_DIR=$(pwd)" \
    -t $USER/weakft:v0.1.2 .
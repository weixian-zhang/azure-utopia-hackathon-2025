
az identity create --resource-group rg-cpf-hackathon-2025 --name identity-aca-acr-cpf-hackathon --location southeastasia

az role assignment create \
--assignee-object-id "fba70817-cff7-4d4e-98f3-2774f1639a42" \
--role "AcrPull" \
--scope "/subscriptions/c9061bc7-fa28-41d9-a783-2600b29c6e2f/resourceGroups/rg-cpf-hackathon-2025/providers/Microsoft.ContainerRegistry/registries/acrcpfhackthon" \
--assignee-principal-type ServicePrincipal

# create container app environment
# az containerapp env create --assign-identity --enable-workload-profiles --resource-group "rg-clamblob" --name "cae-common" --location "southeastasia" --infrastructure-subnet-resource-id /subscriptions/c9061bc7-fa28-41d9-a783-2600b29c6e2f/resourceGroups/rg-clamblob/providers/Microsoft.Network/virtualNetworks/vnet-clamblob/subnets/CAEnvironment --logs-workspace-id c596239e-beda-4999-a79f-58e34d9881e4
#az containerapp env workload-profile set --resource-group rg-clamblob --name cae-common --workload-profile-type D4 --workload-profile-name dedicated-d4-1 --min-nodes 1 --max-nodes 1
az containerapp env create \
    --resource-group rg-cpf-hackathon-2025 \
    --name aca-env-cpf-hackathon \
    --location "southeastasia" \
    --infrastructure-subnet-resource-id /subscriptions/c9061bc7-fa28-41d9-a783-2600b29c6e2f/resourceGroups/rg-cpf-hackathon-2025/providers/Microsoft.Network/virtualNetworks/vnet-cpf-hackathon/subnets/Container-App-Environment \
    --internal-only \
    --enable-peer-to-peer-encryption \
    --logs-workspace-id f2aba6b2-3607-4eb6-aec3-3d52ed5c9d92

# stage_1 streamlit container app
az containerapp create \
--name aca-streamlit \
--resource-group rg-cpf-hackathon-2025 \
--environment /subscriptions/c9061bc7-fa28-41d9-a783-2600b29c6e2f/resourceGroups/rg-cpf-hackathon-2025/providers/Microsoft.App/managedEnvironments/aca-env-cpf-hackathon \
--image acrcpfhackthon.azurecr.io/webapp:v0.0.1 \
--registry-server acrcpfhackthon.azurecr.io \
--registry-identity /subscriptions/c9061bc7-fa28-41d9-a783-2600b29c6e2f/resourceGroups/rg-cpf-hackathon-2025/providers/Microsoft.ManagedIdentity/userAssignedIdentities/identity-aca-acr-cpf-hackathon \
--target-port 8080 \
--ingress external \
--query properties.configuration.fullQdn

# stage_2 hugging face evil llm container app
az containerapp create \
--name aca-evil-llm \
--resource-group rg-cpf-hackathon-2025 \
--environment /subscriptions/c9061bc7-fa28-41d9-a783-2600b29c6e2f/resourceGroups/rg-cpf-hackathon-2025/providers/Microsoft.App/managedEnvironments/aca-env-cpf-hackathon \
--image acrcpfhackthon.azurecr.io/evil-llm:latest \
--registry-server acrcpfhackthon.azurecr.io \
--registry-identity /subscriptions/c9061bc7-fa28-41d9-a783-2600b29c6e2f/resourceGroups/rg-cpf-hackathon-2025/providers/Microsoft.ManagedIdentity/userAssignedIdentities/identity-aca-acr-cpf-hackathon \
--target-port 8080 \
--ingress external \
--query properties.configuration.fullQdn

# mount azure file share to container app environment
# az containerapp env storage set \
#   --access-mode ReadWrite \
#   --azure-file-account-name {$STORAGE_ACCOUNT_NAME} \
#   --azure-file-account-key {$STORAGE_ACCOUNT_KEY} \
#   --azure-file-share-name clamblob-scan \
#   --storage-name clamblob-scan \
#   --name cae-common \
#   --resource-group rg-clamblob \
#   --output table


# create clamav container app
# az containerapp create -n clamblob-clamav-1 -g rg-clamblob --yaml "clamav_container_app.yaml"


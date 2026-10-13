#!/bin/bash

# Script to set environment variables and run the application
# Usage: ./setup_env.sh

# Màu sắc terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===============================================${NC}"
echo -e "${YELLOW}  SETTING UP ENVIRONMENT FOR RAG CHATBOT APPLICATION  ${NC}"
echo -e "${YELLOW}===============================================${NC}"

# Check requirements
echo -e "\n${YELLOW}Checking requirements...${NC}"
command -v python3 >/dev/null 2>&1 || { echo -e "${RED}❌ Python 3 is not installed.${NC}"; exit 1; }
command -v pip3 >/dev/null 2>&1 || { echo -e "${RED}❌ Pip is not installed.${NC}"; exit 1; }

# Install the necessary libraries
echo -e "\n${YELLOW}Installing python-dotenv library...${NC}"
pip3 install python-dotenv

# Create .env file if it does not exist
if [ ! -f .env ]; then
  echo -e "\n${YELLOW}.env file does not exist. Creating new .env file...${NC}"
  cat > .env << EOL
# API key configuration for YeScale.io
OPENAI_API_KEYY=your_yescale_api_key_here
OPENAI_API_KEY_VIP=your_yescale_vip_api_key_here

# Other environment variables
MILVUS_HOST=localhost
MILVUS_PORT=19530
EOL
  echo -e "${GREEN}✅ New .env file created.${NC}"
  echo -e "${YELLOW}⚠️ Please edit the .env file to add your actual API keys.${NC}"
  echo -e "   Type command: nano .env"
else
  echo -e "\n${GREEN}✅ .env file already exists.${NC}"
fi

# Ask the user if they want to update the API key
read -p "Would you like to update the API key now? (y/n): " update_api

if [[ $update_api == "y" || $update_api == "Y" ]]; then
  echo -e "\n${YELLOW}Enter API key information:${NC}"
  read -p "OPENAI_API_KEYY: " api_key
  read -p "OPENAI_API_KEY_VIP: " api_key_vip
  
  # Update .env file
  sed -i.bak "s|OPENAI_API_KEYY=.*|OPENAI_API_KEYY=$api_key|g" .env
  sed -i.bak "s|OPENAI_API_KEY_VIP=.*|OPENAI_API_KEY_VIP=$api_key_vip|g" .env
  rm -f .env.bak
  
  echo -e "${GREEN}✅ API keys updated in the .env file.${NC}"
fi

# Export environment variables
echo -e "\n${YELLOW}Exporting environment variables...${NC}"
export $(grep -v '^#' .env | xargs)

# Notice how to run the application
echo -e "\n${GREEN}✅ Environment setup complete.${NC}"
echo -e "${YELLOW}To run the application, use the command:${NC}"
echo -e "   cd src/app && streamlit run main.py"

# Check API key
echo -e "\n${YELLOW}Checking API key:${NC}"
if [ -z "$OPENAI_API_KEYY" ] || [ "$OPENAI_API_KEYY" == "your_yescale_api_key_here" ]; then
  echo -e "${RED}❌ OPENAI_API_KEYY is not set.${NC}"
else
  echo -e "${GREEN}✅ OPENAI_API_KEYY is set.${NC}"
fi

if [ -z "$OPENAI_API_KEY_VIP" ] || [ "$OPENAI_API_KEY_VIP" == "your_yescale_vip_api_key_here" ]; then
  echo -e "${RED}❌ OPENAI_API_KEY_VIP is not set.${NC}"
else
  echo -e "${GREEN}✅ OPENAI_API_KEY_VIP is set.${NC}"
fi

echo -e "\n${YELLOW}Would you like to run the API key test? (y/n):${NC} "
read run_test

if [[ $run_test == "y" || $run_test == "Y" ]]; then
  echo -e "\n${YELLOW}Running API key test...${NC}"
  python3 test_api.py
fi

echo -e "\n${YELLOW}===============================================${NC}"
echo -e "${GREEN}Done! Thank you for using this tool.${NC}"
echo -e "${YELLOW}===============================================${NC}" 
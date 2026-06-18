#!/bin/bash

# 1. Create a new virtual environment named 'venv'
echo "Creating virtual environment..."
python3 -m venv myenv

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source myenv/bin/activate

# 3. Upgrade pip to the latest version just to be safe
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install all the libraries you saved earlier
echo "Installing libraries from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! To start using your environment, run: source myenv/bin/activate"

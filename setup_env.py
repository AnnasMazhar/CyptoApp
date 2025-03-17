import os
import subprocess
import sys
import venv

def check_and_setup_environment(requirements_file="requirements.txt"):
    """
    Checks for a virtual environment, creates one if necessary,
    activates it, installs dependencies, and checks for missing dependencies.
    """

    env_name = "venv_crypto" # Or your chosen venv name.

    if not os.path.exists(env_name):
        print(f"Creating virtual environment '{env_name}'...")
        venv.create(env_name, with_pip=True)

    if sys.platform == "win32":
        activate_script = os.path.join(env_name, "Scripts", "activate")
        pip_path = os.path.join(env_name, "Scripts", "pip")
    else:
        activate_script = os.path.join(env_name, "bin", "activate")
        pip_path = os.path.join(env_name, "bin", "pip")

    # Activate the virtual environment
    if sys.platform == "win32":
        subprocess.call([activate_script], shell=True) #windows needs shell=True
    else:
        subprocess.call(['source', activate_script], shell=True)

    print(f"Virtual environment '{env_name}' is now active.")

    if os.path.exists(requirements_file):
        print(f"Installing dependencies from '{requirements_file}'...")
        try:
            subprocess.check_call([
                pip_path,
                "install",
                "-r",
                requirements_file,
            ])
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")

        # Check for missing dependencies
        print("Checking for missing dependencies...")
        try:
            installed_packages = subprocess.check_output([pip_path, "freeze"], text=True).splitlines()
            required_packages = []
            with open(requirements_file, 'r') as f:
                required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            installed_names = [pkg.split('==')[0] for pkg in installed_packages]
            missing_packages = [pkg for pkg in required_packages if pkg not in installed_names]

            if missing_packages:
                print(f"Missing packages: {missing_packages}")
                print("Attempting to install missing packages...")
                subprocess.check_call([pip_path, "install"] + missing_packages)
                print("Missing packages installed.")
            else:
                print("All dependencies are satisfied.")
        except subprocess.CalledProcessError as e:
            print(f"Error checking or installing missing dependencies: {e}")

    else:
        print(f"'{requirements_file}' not found. Skipping dependency installation.")
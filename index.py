import sys
import os
import subprocess

def main():
    methods = {
        '1': 'bezier_convex.py',
        '2': 'bezier_alpha.py',
        '3': 'gaussian_shell.py',
        '4': 'bezier_chord.py',
        '5': 'linear_contraction.py'
    }

    print("==================================================")
    print("          AUGMENTATION METHOD SELECTOR           ")
    print("==================================================")
    print("1. Bezier Convex")
    print("2. Bezier Alpha")
    print("3. Gaussian Shell")
    print("4. Bezier Chord")
    print("5. Linear Contraction")
    print("6. Run All Methods")
    print("==================================================")
    
    try:
        choice = input("Select a method to run (1-6) or 'q' to quit: ").strip()
        if choice.lower() == 'q':
            return
            
        dataset_path = input("Enter dataset file path (or press Enter to use default): ").strip()
        
        if choice == '6':
            print("\nExecuting ALL methods sequentially...\n")
            for key, script_name in methods.items():
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
                if os.path.exists(script_path):
                    print(f"\n[{key}/5] Executing {script_name}...\n" + "-"*40)
                    args = [sys.executable, script_path]
                    if dataset_path:
                        args.append(dataset_path)
                    subprocess.run(args)
                else:
                    print(f"\nError: Could not find {script_name} in the current directory.")
            print("\nAll methods have finished execution.")
        elif choice in methods:
            script_name = methods[choice]
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
            
            if os.path.exists(script_path):
                print(f"\nExecuting {script_name}...\n")
                args = [sys.executable, script_path]
                if dataset_path:
                    args.append(dataset_path)
                subprocess.run(args)
            else:
                print(f"\nError: Could not find {script_name} in the current directory.")
        else:
            print("\nInvalid choice. Please enter a number between 1 and 6.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")

if __name__ == "__main__":
    main()

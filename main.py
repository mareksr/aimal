#!/bin/python3

import os
import sys
import time
import shutil
from datetime import datetime
from colorama import Fore, Style, init
from ml_model import train_and_evaluate_classifiers, load_model
import pwd
import grp

init(autoreset=True)

# Define the base path using the environment variable or default to /opt/aimal
AIMAL_PATH = os.getenv('AIMAL_PATH', '/opt/aimal')
GOOD_FILE = os.path.join(AIMAL_PATH, 'good.txt')
MALWARE_DIR = os.path.join(AIMAL_PATH, 'malware')
QUARANTINE_DIR = os.path.join(AIMAL_PATH, 'quarantine')

def get_file_owner_group(file_path):
    stat_info = os.stat(file_path)
    uid = stat_info.st_uid
    gid = stat_info.st_gid
    user = pwd.getpwuid(uid).pw_name
    group = grp.getgrgid(gid).gr_name
    return user, group

def set_file_owner_group(file_path, user, group):
    uid = pwd.getpwnam(user).pw_uid
    gid = grp.getgrnam(group).gr_gid
    os.chown(file_path, uid, gid)

def classify_file_with_options(file_path, model, vectorizer, no_good=False, quarantine=False):
    if not os.path.exists(file_path):
        print("File does not exist.")
        return

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    X = vectorizer.transform([content]).toarray()
    prediction = model.predict(X)

    if prediction[0] == 1:
        print(Fore.RED + f"Malware detected in {file_path}")
        if quarantine:
            if not os.path.exists(QUARANTINE_DIR):
                os.makedirs(QUARANTINE_DIR)
            sanitized_path = file_path.replace('/', '_')
            quarantine_path = os.path.join(QUARANTINE_DIR, sanitized_path)
            shutil.move(file_path, quarantine_path)
            user, group = get_file_owner_group(quarantine_path)
            print(Fore.YELLOW + f"Moved {file_path} to quarantine as {quarantine_path}")
            return file_path, quarantine_path, user, group
    elif not no_good:
        print(Fore.GREEN + f"No malware detected in {file_path}")
    return None

def classify_directory(directory, model, vectorizer, no_good=False, stats=False, quarantine=False):
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    total_files = 0
    malware_files = 0
    start_time = time.time()

    quarantine_report = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.php'):
                total_files += 1
                file_path = os.path.join(root, file)

                result = classify_file_with_options(file_path, model, vectorizer, no_good=no_good, quarantine=quarantine)
                if result:
                    original_path, quarantine_path, user, group = result
                    quarantine_report.append(f"{original_path} : {quarantine_path} : {user} : {group}")
                    malware_files += 1

                print(f"Checked {total_files} files...", end='\r')

    end_time = time.time()
    elapsed_time = end_time - start_time

    if quarantine_report:
        report_filename = os.path.join(QUARANTINE_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_filename, 'w') as report_file:
            report_file.write(f"Scanned Directory: {directory}\n")
            report_file.write("Quarantine Report\n")
            report_file.write("Original Path : Quarantine Path : User : Group\n")
            for line in quarantine_report:
                report_file.write(line + '\n')
            if stats:
                report_file.write("\nStatistics\n")
                report_file.write(f"Total files checked: {total_files}\n")
                report_file.write(f"Malware files: {malware_files}\n")
                report_file.write(f"No malware files: {total_files - malware_files}\n")
                report_file.write(f"Ratio (Malware/Total): {malware_files / total_files:.2f}\n")
                report_file.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
        print(Fore.CYAN + f"Report saved as {report_filename}")

    if stats:
        print(f"\nTotal files checked: {total_files}")
        print(f"Malware files: {malware_files}")
        print(f"No malware files: {total_files - malware_files}")
        print(f"Ratio (Malware/Total): {malware_files / total_files:.2f}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

def restore_files(report_file):
    if not os.path.exists(report_file):
        print("Report file does not exist.")
        return

    with open(report_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if " : " in line:
            original_path, quarantine_path, user, group = line.strip().split(" : ")
            if os.path.exists(quarantine_path):
                original_dir = os.path.dirname(original_path)
                if not os.path.exists(original_dir):
                    os.makedirs(original_dir)
                shutil.move(quarantine_path, original_path)
                set_file_owner_group(original_path, user, group)
                print(Fore.GREEN + f"Restored {quarantine_path} to {original_path} with owner {user} and group {group}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <learn|check|checkdir|restore> [file_to_classify|directory_to_check|report_file] [--no-good] [--stats] [-q]")
        sys.exit(1)

    command = sys.argv[1]
    no_good = "--no-good" in sys.argv
    stats = "--stats" in sys.argv
    quarantine = "-q" in sys.argv

    if command == "learn":
        train_and_evaluate_classifiers(GOOD_FILE, MALWARE_DIR)
    else:
        model, vectorizer = load_model()
        if command == "check":
            if len(sys.argv) < 3:
                print("Usage: python main.py check <file_to_classify> [--no-good] [--stats] [-q]")
                sys.exit(1)
            file_to_classify = sys.argv[2]
            classify_file_with_options(file_to_classify, model, vectorizer, no_good=no_good, quarantine=quarantine)
        elif command == "checkdir":
            if len(sys.argv) < 3:
                print("Usage: python main.py checkdir <directory_to_check> [--no-good] [--stats] [-q]")
                sys.exit(1)
            directory_to_check = sys.argv[2]
            classify_directory(directory_to_check, model, vectorizer, no_good=no_good, stats=stats, quarantine=quarantine)
        elif command == "restore":
            if len(sys.argv) < 3:
                print("Usage: python main.py restore <report_file>")
                sys.exit(1)
            report_file = sys.argv[2]
            restore_files(report_file)
        else:
            print("Unknown command. Use 'learn' to train the model, 'check' to classify a file, 'checkdir' to classify all .php files in a directory, or 'restore' to restore files from a quarantine report.")

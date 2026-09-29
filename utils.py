import re
import os
import subprocess

# 执行git命令
def run_git_command(command):
    """
    Helper function to run a Git command and return the output as a string.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e.stderr)
        return None

# 获取合并节点的两方父节点
def get_parent_hashes(merge_commit_hash):
    parent_hashes_output = run_git_command(f"git log --pretty=%P -n 1 {merge_commit_hash}")
    if not parent_hashes_output:
        return None
    parent_hashes = parent_hashes_output.split()
    if len(parent_hashes) != 2:
        print("Unexpected number of parent hashes found.")
        return None
    return parent_hashes

# 获取两方父节点的公共祖先节点
def get_merge_base(parent1, parent2):
    return run_git_command(f"git merge-base {parent1} {parent2}")

# 获取commit message信息
def get_commit_message(commit_hash):
    commit_message = run_git_command(f"git log -1 --pretty=%B {commit_hash}")
    return commit_message

# 执行-根据工作目录和commit hash获取commit message
def execute(target_directory, merge_commit_hash):
    if not os.path.exists(target_directory):
        print(f"Directory does not exist: {target_directory}")
        return
    
    os.chdir(target_directory)

    # Step 1: Get the two parent hashes of the merge commit
    parent_hashes = get_parent_hashes(merge_commit_hash)
    if not parent_hashes:
        print("Failed to retrieve parent hashes.")
        return
    
    parent1, parent2 = parent_hashes
    
    # Step 2: Get the commit messages for each parent
    parent1_message = get_commit_message(parent1)
    parent2_message = get_commit_message(parent2)

    # Step 3: Get the commit message for the common ancestor
    common_ancestor_hash = get_merge_base(parent1, parent2)
    if not common_ancestor_hash:
        print("Failed to find the common ancestor.")
        return
    common_ancestor_message = get_commit_message(common_ancestor_hash)

     # Git 的合并提交默认会将 当前分支 的最新提交作为第一个父提交，而将 来源分支 的最新提交作为第二个父提交。
    return parent1_message, parent2_message, common_ancestor_message

"""token-level result中的diff 标记都替换为不带文件地址的"""
def rewrite_DiffMarks(input_str):
    # 使用正则表达式进行替换
    output_str = re.sub(r'<<<<<<<.*|>>>>>>>.*|\|\|\|\|\|\|\|.*', 
                        lambda x: '<<<<<<<' if '<<<<<<<' in x.group(0) 
                        else ('>>>>>>>' if '>>>>>>>' in x.group(0) 
                        else '|||||||'), input_str)
    return output_str

def extract_resolved_code(text):
    # 匹配所有用 ``` 包裹的代码块
    matches = re.findall(r'```(?:\w*\s*)?(.*?)(?:```|$)', text, re.DOTALL)
    if matches:
        # 返回最后一个匹配的代码块，并去除首尾空白字符
        return matches[-1].strip()
    return "_None_"

# def extract_resolved_code(text):
#     # 匹配 "Resolved Code:" 后的代码块
#     match = re.search(r'Resolved Code:\s*(```\w*\s*\n)?(.*?)(```|$)', text, re.DOTALL)
#     if match:
#         code = match.group(2).strip()  # 获取代码部分并去除首尾空白字符
#         return code
    
#     # 匹配没有 "Resolved Code:" 的情况
#     match = re.search(r'```\w*\s*\n?(.*?)(```|$)', text, re.DOTALL)
#     if match:
#         code = match.group(1).strip()
#         return code
#     return None

def simplePrompt(conflict):
    return f"""
Task Instructions: Your task is to resolve the given conflict blocks by leveraging extensive experience in resolving code merge conflict.
Merge Conflict to be resolved Next: The merge Conflict block is given below - 
{conflict}.
Resolve the "Merge Conflict to Be Resolved Next" and produce the "Resolved Code" below according to the "Task Instructions". Please enclose the output code with markdown ```.
"""

def prompt_message(message_1, message_2, message_3, conflict):
    return f"""
Task Instructions: Your task is to resolve the given conflict blocks by leveraging extensive experience in resolving code merge conflict and the commit message information from each branch of the provided conflict blocks.
Merge Conflict to be resolved Next: The merge Conflict block is given below - 
{conflict}.
Commit message of the Merge Conflict to be resolved: The commit message of the current branch is - {message_1}, the commit message of the common ancestor node is - {message_2}, and the commit message of the source branch is - {message_3}.
Resolve the "Merge Conflict to Be Resolved Next" and produce the "Resolved Code" below according to the "Task Instructions". Please enclose the output code with markdown ```.
"""

def prompt1_message(reference_c1, reference_r1, message_1, message_2, message_3, conflict):
    return f"""
Task Instructions: Your task is to resolve the given conflict blocks by leveraging extensive experience in resolving code merge conflict, the commit message information from each branch of the given conflict blocks and the provided examples of merge conflict resolutions in the same code repository.
Reference example of merge conflict resolution in the same code repository: 
Reference 1. The similar merge conflict is - 
{reference_c1}
; and Reference 1 its resolution is - 
{reference_r1}.
Merge Conflict to be resolved Next: The merge Conflict block is given below - 
{conflict}.
Commit message of the Merge Conflict to be resolved: The commit message of the current branch is - {message_1}, the commit message of the common ancestor node is - {message_2}, and the commit message of the source branch is - {message_3}.
Resolve the "Merge Conflict to Be Resolved Next" and produce the "Resolved Code" below according to the "Task Instructions", the "Commit message of the Merge Conflict to be resolved" and the resolution pattern from the "Reference Example of Merge Conflict Resolution in the Same Code Repository". When the content of the conflict blocks is completely identical, please strictly follow the resolution pattern provided in the reference example. Please enclose the output code with markdown ```.
"""

def prompt1(reference_c1, reference_r1, conflict):
    return f"""
Task Instructions: Your task is to resolve the given conflict blocks by leveraging extensive experience in resolving code merge conflict and the provided examples of merge conflict resolutions in the same code repository.
Reference example of merge conflict resolution in the same code repository: 
Reference 1. The similar merge conflict is - 
{reference_c1}
; and Reference 1 its resolution is - 
{reference_r1}.
Merge Conflict to be resolved Next: The merge Conflict block is given below - 
{conflict}.
Resolve the "Merge Conflict to Be Resolved Next" and produce the "Resolved Code" below according to the "Task Instructions" and the resolution pattern from the "Reference Example of Merge Conflict Resolution in the Same Code Repository". When the content of the conflict blocks is completely identical, please strictly follow the resolution pattern provided in the reference example. Please enclose the output code with markdown ```.
"""

def prompt2(reference_c1, reference_r1, reference_c2, reference_r2, conflict):
    return f"""
Task Instructions: Your task is to resolve the given conflict blocks by leveraging extensive experience in resolving code merge conflict and the provided examples of merge conflict resolutions in the same code repository.
Reference example of merge conflict resolution in the same code repository: 
Reference 1. The similar merge conflict is - 
{reference_c1}
; and Reference 1 its resolution is - 
{reference_r1}.
Reference 2. The similar merge conflict is - 
{reference_c2}
; and Reference 2 its resolution is - 
{reference_r2}.
Merge Conflict to be resolved Next: The merge Conflict block is given below - 
{conflict}.
Resolve the "Merge Conflict to Be Resolved Next" and produce the "Resolved Code" below according to the "Task Instructions" and the resolution pattern from the "Reference Example of Merge Conflict Resolution in the Same Code Repository". When the content of the conflict blocks is completely identical, please strictly follow the resolution pattern provided in the reference example. Please enclose the output code with markdown ```.
"""

def prompt3(reference_c1, reference_r1, reference_c2, reference_r2, reference_c3, reference_r3, conflict):
    return f"""
Task Instructions: Your task is to resolve the given conflict blocks by leveraging extensive experience in resolving code merge conflict and the provided examples of merge conflict resolutions in the same code repository.
Reference example of merge conflict resolution in the same code repository: 
Reference 1. The similar merge conflict is - 
{reference_c1}
; and Reference 1 its resolution is - 
{reference_r1}.
Reference 2. The similar merge conflict is - 
{reference_c2}
; and Reference 2 its resolution is - 
{reference_r2}.
Reference 3. The similar merge conflict is - 
{reference_c3}
; and Reference 3 its resolution is - 
{reference_r3}.
Merge Conflict to be resolved Next: The merge Conflict block is given below - 
{conflict}.
Resolve the "Merge Conflict to Be Resolved Next" and produce the "Resolved Code" below according to the "Task Instructions" and the resolution pattern from the "Reference Example of Merge Conflict Resolution in the Same Code Repository". When the content of the conflict blocks is completely identical, please strictly follow the resolution pattern provided in the reference example. Please enclose the output code with markdown ```.
"""

def prompt4(reference_c1, reference_r1, reference_c2, reference_r2, reference_c3, reference_r3, reference_c4, reference_r4, conflict):
    return f"""
Task Instructions: Your task is to resolve the given conflict blocks by leveraging extensive experience in resolving code merge conflict and the provided examples of merge conflict resolutions in the same code repository.
Reference example of merge conflict resolution in the same code repository: 
Reference 1. The similar merge conflict is - 
{reference_c1}
; and Reference 1 its resolution is - 
{reference_r1}.
Reference 2. The similar merge conflict is - 
{reference_c2}
; and Reference 2 its resolution is - 
{reference_r2}.
Reference 3. The similar merge conflict is - 
{reference_c3}
; and Reference 3 its resolution is - 
{reference_r3}.
Reference 4. The similar merge conflict is - 
{reference_c4}
; and Reference 4 its resolution is - 
{reference_r4}.
Merge Conflict to be resolved Next: The merge Conflict block is given below - 
{conflict}.
Resolve the "Merge Conflict to Be Resolved Next" and produce the "Resolved Code" below according to the "Task Instructions" and the resolution pattern from the "Reference Example of Merge Conflict Resolution in the Same Code Repository". When the content of the conflict blocks is completely identical, please strictly follow the resolution pattern provided in the reference example. Please enclose the output code with markdown ```.
"""

def prompt5(reference_c1, reference_r1, reference_c2, reference_r2, reference_c3, reference_r3, reference_c4, reference_r4, reference_c5, reference_r5, conflict):
    return f"""
Task Instructions: Your task is to resolve the given conflict blocks by leveraging extensive experience in resolving code merge conflict and the provided examples of merge conflict resolutions in the same code repository.
Reference example of merge conflict resolution in the same code repository: 
Reference 1. The similar merge conflict is - 
{reference_c1}
; and Reference 1 its resolution is - 
{reference_r1}.
Reference 2. The similar merge conflict is - 
{reference_c2}
; and Reference 2 its resolution is - 
{reference_r2}.
Reference 3. The similar merge conflict is - 
{reference_c3}
; and Reference 3 its resolution is - 
{reference_r3}.
Reference 4. The similar merge conflict is - 
{reference_c4}
; and Reference 4 its resolution is - 
{reference_r4}.
Reference 5. The similar merge conflict is - 
{reference_c5}
; and Reference 5 its resolution is - 
{reference_r5}.
Merge Conflict to be resolved Next: The merge Conflict block is given below - 
{conflict}.
Resolve the "Merge Conflict to Be Resolved Next" and produce the "Resolved Code" below according to the "Task Instructions" and the resolution pattern from the "Reference Example of Merge Conflict Resolution in the Same Code Repository". When the content of the conflict blocks is completely identical, please strictly follow the resolution pattern provided in the reference example. Please enclose the output code with markdown ```.
"""
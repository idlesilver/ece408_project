import os
import re
import time
import wget
current_op = "op2"

root_dir = "/home/gavin/ece408/ece408_project"
summary_file = os.path.join("/home/gavin/ece408/build/m3",current_op,"summary.md")
summary_dir = os.path.join("/home/gavin/ece408/build/m3",current_op)

commands = {
    "100":'- /bin/bash -c "time ./m3 100"',
    "1000":'- /bin/bash -c "time ./m3 1000"',
    "10000":'- /bin/bash -c "time ./m3 10000"',
    "nsys":'- nsys profile --stats=true ./m3 100',
    "analysis-file":"- nv-nsight-cu-cli --section '.*' -o analysis-file ./m3 100",
}


os.chdir(root_dir)

with open("custom/new-forward-"+current_op+".cu", "r") as f:
    op_file = f.readlines()

with open("custom/new-forward.cu","w") as f:
    f.writelines(op_file)

with open("rai_build_tamplate.yml","r") as f:
    template = f.readlines()

os.system(f'echo "\n# {current_op}" >> {summary_file}')
os.system(f'echo "{time.strftime("%a %b %d %H:%M:%S", time.localtime())}" >> {summary_file}')
os.system(f'echo "commands: {[key for key in commands.keys()]}" >> {summary_file}')
for key in commands:
    with open("rai_build.yml","w") as f:
        f.writelines(template)
        f.write("      "+commands[key])
    os.system(f'echo "## {key}" >> {summary_file}')
    os.system(f'echo "\`\`\`sh" >> {summary_file}')
    
    # 执行命令，输出到对应command key文件 
    try:
        output_file = os.path.join(summary_dir,key+".txt")
        os.system(f'rai -p ./ --queue rai_amd64_exclusive > {output_file} 2>&1')
        
        # 把有用的信息提取出来
        with open(output_file,"r") as f:
            output = f.readlines()
        result = []
        success = False
        for line in output[::-1]:
            if re.search(f"{' '.join(commands[key].split()[-3:])}",line):
                result.append(line)
                success = True
                break
            if re.search(f"^Importing",line):
                continue
            result.append(line)
        result
        with open(summary_file,"a") as f:
            if success:
                f.writelines(result[::-1])
            else:
                f.write("fail")
    except Exception as e:
        print(e)
    finally:
        os.system(f'echo "\`\`\`\n" >> {summary_file}')

    if key=="analysis-file":
        url = re.match(".+(http://s3.amazonaws.com/files.rai-project.com/userdata/build-\S+\.tar\.gz)",result[0]).group(1)
        build_tar = os.path.join(summary_dir,'analysis-file.tar.gz')
        # os.system(f"wget {url} -o {build_tar}")
        wget.download(url,build_tar)
    else:
        time.sleep(60*2.5)
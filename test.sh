
#!/bin/bash
for filename in garagePems/*.pth; do
python test.py --checkpoint $filename
done


import sys

srcFile=open(sys.argv[1])
tgtFileAttn=open(sys.argv[2]+".output")
tgtFileNonAttn=open(sys.argv[3]+".output")
timeFileAttn=open(sys.argv[2]+".time")
timeFileNonAttn=open(sys.argv[3]+".time")

L=14
C=60

srcLines=srcFile.readlines()
attnLines=tgtFileAttn.readlines()
nonAttnLines=tgtFileNonAttn.readlines()
timeAttnLines=[float(line.split()[0]) for line in timeFileAttn.readlines()]
timeNonAttnLines=[float(line.split()[0]) for line in timeFileNonAttn.readlines()]

#print len(srcLines)
#print len(attnLines)
#print len(nonAttnLines)

finalOutputFile=open("mixed.out","w")
finalLines=[]

cAttn=sum(timeAttnLines)
cNonAttn=sum(timeNonAttnLines)
cMixed=0

for i,line in enumerate(srcLines):
    words=line.split()
    charLength=sum([len(word) for word in words])
    #if len(words)<L:
    if charLength<C:
        finalLines.append(nonAttnLines[i])
        cMixed+=timeNonAttnLines[i]
    else:
        finalLines.append(attnLines[i])
        cMixed+=timeAttnLines[i]

for line in finalLines:
    finalOutputFile.write(line)

finalOutputFile.close()
print cAttn
print cNonAttn
print cMixed

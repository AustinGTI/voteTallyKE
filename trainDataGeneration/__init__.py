import utilityFunctions,os,cv2


def generateDataset(n,generator,folder,noiseMaps,templatePaths,extraData = [],split=[0.8,0.15,0.05]):
    # set up locations
    reqPaths = [[folder], ["images", "labels"], ["trainSet", "valSet", "testSet"]]
    for root in reqPaths[0]:
        if not os.path.exists(root):
            os.mkdir(root)
        for mid in reqPaths[1]:
            if not os.path.exists(os.path.join(root, mid)):
                os.mkdir(os.path.join(root, mid))
            for leaf in reqPaths[2]:
                if not os.path.exists(os.path.join(root, mid, leaf)):
                    os.mkdir(os.path.join(root, mid, leaf))

    counts = [0, 0, 0]
    noiseMapSets = [utilityFunctions.createNoiseMaps(*nm) for nm in noiseMaps]
    #noiseMaps = utilityFunctions.createNoiseMaps(50, 700, 700)  # change this if the size of the imgs


    for b in range(n):
        for si in range(len(split)):
            if b / n < sum(split[:si + 1]):
                thisSet = reqPaths[2][si]
                break

        if b % (n // 10) == 0:
            print(f"Created {b} out of {n} samples of training data")
        # im,bounds = createDigitGrid(chars)
        templateIms = [cv2.imread(path,cv2.IMREAD_GRAYSCALE) for path in templatePaths]
        im, bounds = generator(noiseMapSets,templateIms,extraData)
        impath = f"{reqPaths[0][0]}/{reqPaths[1][0]}/{thisSet}/{utilityFunctions.padVal(counts[si] + 1, 4)}.jpg"
        lbpath = f"{reqPaths[0][0]}/{reqPaths[1][1]}/{thisSet}/{utilityFunctions.padVal(counts[si] + 1, 4)}.txt"
        cv2.imwrite(impath, cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
        with open(lbpath, "w") as file:
            for bd in bounds:
                file.write(bd)
        counts[si] += 1
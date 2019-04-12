import math

class spatialHashTable(object):
    def __repr__(self):
        name = "Hash table: \n"
        name += "Rows: " + str(self.rows) + " Cols: " + str(self.cols) + " Cellsize: " + str(self.bucketSize) + "\n"
        #name += "Hash table content: \n"
        total = 0
        for bucketId, content in self.buckets.items():
        #    name += "Bucket: " + str(bucketId) + " contains " + str(len(content)) + " items." + "\n"
            total += len(content)
        name += "Total objects in table: " + str(total)
        return name

    def __init__(self, hashTableSize, bucketSize, left = 0, top = 0):
        self.left = left
        self.top = top
        self.size = hashTableSize
        self.rows = int(math.ceil(hashTableSize / bucketSize))
        self.cols = self.rows
        self.bucketSize = bucketSize
        self.buckets = {}
        self.clearBuckets()

    def getNearbyObjects(self, obj):
        cellIds = self.getIdsForObj(obj)
        return self.getObjectsFromBuckets(cellIds)

    def getNearbyObjectsInArea(self, pos, rad):
        cellIds = self.getIdsForArea(pos, rad)
        return self.getObjectsFromBuckets(cellIds)

    def getNearbyEnemyObjects(self, obj):
        cellIds = self.getIdsForObj(obj)
        nearbyObjects = self.getObjectsFromBuckets(cellIds)
        return [nearbyObject for nearbyObject in nearbyObjects if nearbyObject.getPlayer() is not obj.getPlayer()]

    def getObjectsFromBuckets(self, cellIds):
        nearbyObjects = set()
        for cellId in cellIds:
            for cell in self.buckets[cellId]:
                nearbyObjects.add(cell)
        return nearbyObjects

    def clearBuckets(self):
        for i in range(self.cols * self.rows):
            self.buckets[i] = []

    def insertObject(self, obj):
        cellIds = self.getIdsForObj(obj)
        for id in cellIds:
            self.buckets[id].append(obj)

    def insertAllObjects(self, objects):
        for obj in objects:
            self.insertObject(obj)

    # Deletes an object out of all the buckets it is in. Might not be needed as it might
    # be faster to clear all buckets and reinsert items than updating objects.
    def deleteObject(self, obj):
        cellIds = self.getIdsForObj(obj)
        for id in cellIds:
            self.buckets[id].remove(obj)

    def getIdsForObj(self, obj):
        pos = obj.getPos()
        radius = obj.getRadius()
        return self.getIdsForArea(pos, radius)

    def getIdsForArea(self, pos, radius):
        ids = set()
        hashFunc = self.getHashId
        cellLeft = max(0, pos[0] - radius)
        cellTop = max(0, pos[1] - radius)
        bucketLeft = int(cellLeft - cellLeft % self.bucketSize)
        bucketTop = int(cellTop - cellTop % self.bucketSize)
        limitX = int(min(self.size, pos[0] + radius + 1))
        limitY = int(min(self.size, pos[1] + radius + 1))

        for x in range(bucketLeft, limitX, self.bucketSize):
            for y in range(bucketTop, limitY, self.bucketSize):
                ids.add(hashFunc(x, y))
        return ids

    def insertAllFloatingPointObjects(self, objects):
        for obj in objects:
            ids = self.getIdsForAreaFloatingPoint(obj.getPos(), obj.getRadius())
            for id in ids:
                self.buckets[id].append(obj)

    def getIdsForAreaFloatingPoint(self, pos, radius):
        ids = set()
        hashFunc = self.getHashId
        pos = (pos[0] - self.left, pos[1] - self.top)
        cellLeft = max(0, pos[0] - radius)
        cellTop = max(0, pos[1] - radius)
        bucketLeft = cellLeft - cellLeft % self.bucketSize
        bucketTop = cellTop - cellTop % self.bucketSize
        limitX = min(self.size - 1, pos[0] + radius)
        limitY = min(self.size - 1, pos[1] + radius)
        x = bucketLeft
        while x <= limitX:
            y = bucketTop
            while y <= limitY:
                ids.add(hashFunc(x,y))
                y += self.bucketSize
            x += self.bucketSize
        return ids


    def getHashId(self, x, y):
        return int(x / self.bucketSize) + int(y / self.bucketSize) * self.cols



    def getCols(self):
        return self.cols

    def getRows(self):
        return self.rows

    def getBuckets(self):
        return self.buckets

    def getBucketContent(self, idx):
        return self.buckets[idx]

    def getCenterOfBucket(self, id):
        x = id % self.cols * self.bucketSize + self.bucketSize / 2
        y = int(id /self.cols) * self.bucketSize + self.bucketSize / 2
        return (x,y)

    # Currently not in use
    def getCenterOfNextEmptyBucket(self, pos):
        startId = self.getHashId(pos)
        numOfBuckets = self.rows * self.cols

        for i in range(numOfBuckets):
            if not bool(self.getObjectsFromBucket((startId + i) % numOfBuckets)):
                return self.getCenterOfBucket((startId + i) % numOfBuckets)
        return self.getCenterOfBucket(startId)

        #[(return self.getObjectsFromBuckets(self.getHashId(pos))]

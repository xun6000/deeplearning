from PIL import Image


numberoftest = 100
numberofclass = 10
imsize = 28





# for i in range(0, numberofclass):
#     for j in range(0, numberoftest):
#         path = 'CIFAR10/Test/%d/Image%05d.png' % (i, j)
#         img = Image.open(path)
#         img.transpose(Image.FLIP_LEFT_RIGHT).save('CIFAR10/lr/%d/Image%05d.png' % (i, j))
for i in range(0, numberofclass):
    for j in range(0, numberoftest):
        path = 'CIFAR10/ud/%d/Image%05d.png' % (i, j)
        img = Image.open(path)
        img.transpose(Image.FLIP_TOP_BOTTOM).save('CIFAR10/ud/%d/Image%05d.png' % (i, j))


# for i in range(0, numberofclass):
#     for j in range(0, numberoftest):
#         path = 'CIFAR10/Test/%d/Image%05d.png' % (i, j)
#         img = Image.open(path)
#         img.transpose(Image.ROTATE_90).save('CIFAR10/rotate/%d/Image%05d.png' % (i, j))

print('FINISHED ROTATE')

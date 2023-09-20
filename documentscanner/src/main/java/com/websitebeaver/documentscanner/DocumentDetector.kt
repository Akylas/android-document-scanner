package com.websitebeaver.documentscanner

import android.graphics.*
import com.websitebeaver.documentscanner.extensions.distance
import com.websitebeaver.documentscanner.models.Document
import com.websitebeaver.documentscanner.utils.ImageUtil
import java.io.File
import java.util.Vector

/**
 * This class uses OpenCV to find document corners.
 *
 * @constructor creates document detector
 */
class DocumentDetector {
    companion object {
        private external fun nativeScan(srcBitmap: Bitmap, shrunkImageHeight: Int, imageRotation: Int): Vector<Vector<Point>>
        private external fun nativeCrop(srcBitmap: Bitmap, points: Array<Point>, outBitmap: Bitmap)
        init {
            try {
                System.loadLibrary("document_detector")
            } catch (exception: Exception) {}
        }

        /**
         * take a photo with a document, and find the document's corners
         *
         * @param image a photo with a document
         * @return a list with document corners (top left, top right, bottom right, bottom left)
         */
        fun findDocumentCorners(image: Bitmap, shrunkImageHeight: Double = 500.0, imageRotation: Int= 0): List<List<Point>>? {
            val outPoints =  nativeScan(image, shrunkImageHeight.toInt(), imageRotation)
            if (outPoints.size > 0) {
                if (outPoints[0].size == 0) {
                    return null
                }
                return (outPoints)
                return (outPoints)
            }
            return null
            // convert bitmap to OpenCV matrix
//        val mat = Mat()
//        Utils.bitmapToMat(image, mat)
//
//        // shrink photo to make it easier to find document corners
//        Imgproc.resize(
//            mat,
//            mat,
//            Size(
//                shrunkImageHeight * image.width / image.height,
//                shrunkImageHeight
//            )
//        )
//        if (imageRotation != 0) {
//            when (imageRotation) {
//                90 -> Core.rotate(mat, mat, Core.ROTATE_90_CLOCKWISE);
//                180 -> Core.rotate(mat, mat, Core.ROTATE_180);
//                else -> Core.rotate(mat, mat, Core.ROTATE_90_COUNTERCLOCKWISE);
//            }
//        }
//
//        // convert photo to LUV colorspace to avoid glares caused by lights
//        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2Luv)
//
//        // separate photo into 3 parts, (L, U, and V)
//        val imageSplitByColorChannel: List<Mat> = mutableListOf()
//        Core.split(mat, imageSplitByColorChannel)
//
//        // find corners for each color channel, then pick the quad with the largest
//        // area, and scale point to account for shrinking image before document detection
//        val documentCorners: List<Point>? = imageSplitByColorChannel
//            .mapNotNull { findCorners(it) }
//            .maxByOrNull { Imgproc.contourArea(it) }
//            ?.toList()
//            ?.map {
//                Point(
//                    it.x * image.height / shrunkImageHeight,
//                    it.y * image.height / shrunkImageHeight
//                )
//            }
//        mat.release()
//        // sort points to force this order (top left, top right, bottom left, bottom right)
//        return documentCorners
//            ?.sortedBy { it.y }
//            ?.chunked(2)
//            ?.map { it.sortedBy { point -> point.x } }
//            ?.flatten()
        }

        /**
         * take an image matrix with a document, and find the document's corners
         *
         * @param image a photo with a document in matrix format (only 1 color space)
         * @return a matrix with document corners or null if we can't find corners
         */
//    private fun findCorners(image: Mat): MatOfPoint? {
//        val outputImage = Mat()
//
//        // blur image to help remove noise
//        Imgproc.GaussianBlur(image, outputImage, Size(5.0, 5.0),0.0)
//
//        // convert all pixels to either black or white (document should be black after this), but
//        // there might be other parts of the photo that turn black
//        Imgproc.threshold(
//            outputImage,
//            outputImage,
//            0.0,
//            255.0,
//            Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU
//        )
//
//        // detect the document's border using the Canny edge detection algorithm
//        Imgproc.Canny(outputImage, outputImage, 50.0, 200.0)
//
//        // the detect edges might have gaps, so try to close those
//        Imgproc.morphologyEx(
//            outputImage,
//            outputImage,
//            Imgproc.MORPH_CLOSE,
//            Mat.ones(Size(5.0, 5.0), CvType.CV_8U)
//        )
//
//        // get outline of document edges, and outlines of other shapes in photo
//        val contours: MutableList<MatOfPoint> = mutableListOf()
//        Imgproc.findContours(
//            outputImage,
//            contours,
//            Mat(),
//            Imgproc.RETR_LIST,
//            Imgproc.CHAIN_APPROX_SIMPLE
//        )
//
//        // approximate outlines using polygons
//        var approxContours = contours.map {
//            val approxContour = MatOfPoint2f()
//            val contour2f = MatOfPoint2f(*it.toArray())
//            Imgproc.approxPolyDP(
//                contour2f,
//                approxContour,
//                0.02 * Imgproc.arcLength(contour2f, true),
//                true
//            )
//            MatOfPoint(*approxContour.toArray())
//        }
//
//        // We now have many polygons, so remove polygons that don't have 4 sides since we
//        // know the document has 4 sides. Calculate areas for all remaining polygons, and
//        // remove polygons with small areas. We assume that the document takes up a large portion
//        // of the photo. Remove polygons that aren't convex since a document can't be convex.
//        approxContours = approxContours.filter {
//            it.height() == 4 && Imgproc.contourArea(it) > 1000 && Imgproc.isContourConvex(it)
//        }
//        outputImage.release()
//        // Once we have all large, convex, 4-sided polygons find and return the 1 with the
//        // largest area
//        return approxContours.maxByOrNull { Imgproc.contourArea(it) }
//    }


        /**
         * take a photo with a document, crop everything out but document, and force it to display
         * as a rectangle
         *
         * @param document with original image data
         * @param colorFilter for this image
         * @return bitmap with cropped and warped document
         */
        fun cropDocument(document: Document, colorFilter: ColorFilter?): Bitmap {
            val file = File(document.originalPhotoPath)
            val bitmap = ImageUtil.getImageFromFile(file, 4000)

            // read image with OpenCV
//        val image = Mat()
//        Utils.bitmapToMat(bitmap, image)
//        bitmap.recycle()
//
            // convert corners from image preview coordinates to original photo coordinates
            // (original image is probably bigger than the preview image)
            val preview = document.preview
            val corners =if (preview != null)  document.quad!!.mapPreviewToOriginalImageCoordinates(
                RectF(0f, 0f, 1f * preview.width, 1f * preview.height),
                1f * preview.height / bitmap.height
            ) else document.quad!!
//        // convert top left, top right, bottom right, and bottom left document corners from
//        // Android points to OpenCV points
//        val tLC = corners.topLeftCorner.toOpenCVPoint()
//        val tRC = corners.topRightCorner.toOpenCVPoint()
//        val bRC = corners.bottomRightCorner.toOpenCVPoint()
//        val bLC = corners.bottomLeftCorner.toOpenCVPoint()
//
//        // Calculate the document edge distances. The user might take a skewed photo of the
//        // document, so the top left corner to top right corner distance might not be the same
//        // as the bottom left to bottom right corner. We could take an average of the 2, but
//        // this takes the smaller of the 2. It does the same for height.
//        val width = min(tLC.distance(tRC), bLC.distance(bRC))
//        val height = min(tLC.distance(bLC), tRC.distance(bRC))
//
//        // create empty image matrix with cropped and warped document width and height
//        val croppedImage = MatOfPoint2f(
//            Point(0.0, 0.0),
//            Point(width, 0.0),
//            Point(width, height),
//            Point(0.0, height),
//        )
//
//        // This crops the document out of the rest of the photo. Since the user might take a
//        // skewed photo instead of a straight on photo, the document might be rotated and
//        // skewed. This corrects that problem. output is an image matrix that contains the
//        // corrected image after this fix.
//        val output = Mat()
//        Imgproc.warpPerspective(
//            image,
//            output,
//            Imgproc.getPerspectiveTransform(
//                MatOfPoint2f(tLC, tRC, bRC, bLC),
//                croppedImage
//            ),
//            Size(width, height)
//        )
            // convert output image matrix to bitmap
            val cropWidth = ((corners.topLeftCorner.distance(corners.topRightCorner)
                    + corners.bottomLeftCorner.distance(corners.bottomRightCorner)) / 2) as Int
            val cropHeight = ((corners.bottomLeftCorner.distance(corners.topLeftCorner)
                    + corners.bottomRightCorner.distance(corners.topRightCorner)) / 2) as Int

            val cropBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Bitmap.Config.ARGB_8888)
            nativeCrop(bitmap,  corners.cornersList, cropBitmap)
//        Utils.matToBitmap(output, croppedBitmap)
//
            if (colorFilter != null) {
                val canvas = Canvas(cropBitmap)
                val paint = Paint()
                paint.colorFilter = colorFilter
                canvas.drawBitmap(cropBitmap, 0f, 0f, paint)
            }
//        output.release()
//        image.release()
            return cropBitmap
        }
    }

}
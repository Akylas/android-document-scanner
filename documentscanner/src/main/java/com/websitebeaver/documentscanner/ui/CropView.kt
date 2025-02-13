package com.websitebeaver.documentscanner.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.websitebeaver.documentscanner.R
import com.websitebeaver.documentscanner.extensions.drawQuadSimple
import com.websitebeaver.documentscanner.models.Quad

class CropView(context: Context, attrs: AttributeSet) : View(context, attrs) {

    var imageWidth: Int = 0
    var imageHeight: Int = 0

    /**
     * @property imagePreviewHeight this is needed because height doesn't update immediately
     * after we set the image
     */
    private var imagePreviewHeight = height

    /**
     * @property imagePreviewWidth this is needed because width doesn't update immediately
     * after we set the image
     */
    private var imagePreviewWidth = width
    /**
     * @property imagePreviewBounds image coordinates - if the image ratio is different than
     * the image container ratio then there's blank space either at the top and bottom of the
     * image or the left and right of the image
     */
    val imagePreviewBounds: RectF
        get() {
            // image container width to height ratio
            val imageViewRatio: Float = imagePreviewWidth.toFloat() / imagePreviewHeight.toFloat()

            // image width to height ratio
            val imageRatio = width.toFloat() / height.toFloat()

            var left = 0f
            var top = 0f
            var right = imagePreviewWidth.toFloat()
            var bottom = imagePreviewHeight.toFloat()

            if (imageRatio > imageViewRatio) {
                // if the image is really wide, there's blank space at the top and bottom
                val offset = (imagePreviewHeight - (imagePreviewWidth / imageRatio)) / 2
                top += offset
                bottom -= offset
            } else {
                // if the image is really tall, there's blank space at the left and right
                // it's also possible that the image ratio matches the image container ratio
                // in which case there's no blank space
                val offset = (imagePreviewWidth - (imagePreviewHeight * imageRatio)) / 2
                left += offset
                right -= offset
            }

            return RectF(left, top, right, bottom)
        }
    /**
     * Initially the image preview height is 0. This calculates the height by using the photo
     * dimensions. It maintains the photo aspect ratio (we likely need to scale the photo down
     * to fit the preview container).
     *
     * @param photo the original photo with a rectangular document
     * @param screenWidth the device width
     */
    fun setImagePreviewBounds(photoWidth: Int, photoHeight: Int, screenWidth: Int, screenHeight: Int) {
        // image width to height aspect ratio
        val imageRatio = photoWidth.toFloat() / photoHeight.toFloat()

        imagePreviewHeight = if (photoHeight > photoWidth) {
            // if user takes the photo in portrait
            (screenWidth.toFloat() / imageRatio).toInt()
        } else {
            // if user takes the photo in landscape
            // if user takes the photo in landscape
            (screenWidth.toFloat() * imageRatio).toInt()
        }
        imagePreviewWidth = screenWidth
    }
    /**
     * @property quad the 4 document corners
     */
    public var quads: List<Quad>? = null

    /**
     * @property cropperLinesAndCornersStyles paint style for 4 corners and connecting lines
     */
    private val cropperLinesAndCornersStyles = Paint(Paint.ANTI_ALIAS_FLAG)

    var colors = listOf<Int>(0xFF007AFF.toInt())

    init {
        // set cropper style
        cropperLinesAndCornersStyles.style = Paint.Style.STROKE
        cropperLinesAndCornersStyles.strokeWidth = resources.displayMetrics.density * 2
        setWillNotDraw(false)
    }

    /**
     * This gets called constantly, and we use it to update the cropper corners
     *
     * @param canvas the image preview canvas
     */
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (quads !== null) {
//            val ratio = canvas.width / imageWidth.toFloat()
//            canvas.scale(ratio, ratio)
//            cropperLinesAndCornersStyles.strokeWidth = resources.displayMetrics.density * 2 / ratio
            quads!!.forEachIndexed { index, quad ->
                cropperLinesAndCornersStyles.color = colors[index.mod(colors.size)]
                // draw 4 corners and connecting lines
                canvas.drawQuadSimple(
                    quad!!,
                    cropperLinesAndCornersStyles
                )

            }
        }

    }
}

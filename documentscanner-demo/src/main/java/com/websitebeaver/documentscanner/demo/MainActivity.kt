package com.websitebeaver.documentscanner.demo

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.websitebeaver.documentscanner.DocumentScanner
import com.websitebeaver.documentscanner.extensions.onClick
import com.websitebeaver.documentscanner.utils.ImageUtil

/**
 * A demo showing how to use the document scanner
 *
 * @constructor creates demo activity
 */
class MainActivity : AppCompatActivity() {
    /**
     * @property croppedImageView the cropped image view
     */
    private lateinit var croppedImageView: ImageView
    private lateinit var button: Button

    /**
     * @property documentScanner the document scanner
     */
    private val documentScanner = DocumentScanner(
        this,
        { croppedImageResults ->
            // display the first cropped image
            croppedImageView.setImageBitmap(
                ImageUtil.readBitmapFromFileUriString(
                    croppedImageResults.first(),
                    contentResolver
                )
            )
        },
        {
            // an error happened
                errorMessage -> Log.v("documentscannerlogs", errorMessage)
        },
        {
            // user canceled document scan
            Log.v("documentscannerlogs", "User canceled document scan")
        }
    )
    private fun onButtonClick() {
        documentScanner.startScan()
    }

    /**
     * called when activity is created
     *
     * @param savedInstanceState persisted data that maintains state
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // cropped image
        croppedImageView = findViewById(R.id.cropped_image_view)
        button = findViewById(R.id.scan_button)
        button.setOnClickListener() { onButtonClick() }

    }
}
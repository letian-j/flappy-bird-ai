import cv2
import numpy as np

def template_match(source_image, template_image, threshold=0.8):
    """
    Perform template matching to find the location of a template in a source image.

    Args:
        source_image: The source image (numpy array or path to image)
        template_image: The template image to search for (numpy array or path to image)
        threshold: Minimum correlation coefficient for a match (0-1)

    Returns:
        tuple: (best_match_location, max_correlation_value) or None if no match found
    """
    # Load images if paths are provided
    if isinstance(source_image, str):
        source = cv2.imread(source_image, cv2.IMREAD_GRAYSCALE)
    else:
        source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY) if len(source_image.shape) == 3 else source_image

    if isinstance(template_image, str):
        template = cv2.imread(template_image, cv2.IMREAD_GRAYSCALE)
    else:
        template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY) if len(template_image.shape) == 3 else template_image

    if source is None or template is None:
        raise ValueError("Could not load image(s)")

    # Get dimensions
    h, w = template.shape

    # Perform template matching
    result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)

    # Find the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        return max_loc, max_val
    else:
        return None

def multi_template_match(source_image, template_image, threshold=0.8):
    """
    Find all matches of a template in a source image above a threshold.

    Args:
        source_image: The source image (numpy array or path to image)
        template_image: The template image to search for (numpy array or path to image)
        threshold: Minimum correlation coefficient for a match (0-1)

    Returns:
        list: List of tuples (location, correlation_value) for all matches
    """
    # Load images if paths are provided
    if isinstance(source_image, str):
        source = cv2.imread(source_image, cv2.IMREAD_GRAYSCALE)
    else:
        source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY) if len(source_image.shape) == 3 else source_image

    if isinstance(template_image, str):
        template = cv2.imread(template_image, cv2.IMREAD_GRAYSCALE)
    else:
        template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY) if len(template_image.shape) == 3 else template_image

    if source is None or template is None:
        raise ValueError("Could not load image(s)")

    # Get dimensions
    h, w = template.shape

    # Perform template matching
    result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)

    # Find all locations above threshold
    locations = np.where(result >= threshold)
    matches = []

    for pt in zip(*locations[::-1]):
        matches.append((pt, result[pt[1], pt[0]]))

    # Sort by correlation value (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)

    return matches

def preprocess_image(image, resize_factor=1.0):
    """
    Preprocess image for better template matching.
    Convert to grayscale and optionally resize.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize if needed
    if resize_factor != 1.0:
        h, w = gray.shape
        new_h, new_w = int(h * resize_factor), int(w * resize_factor)
        gray = cv2.resize(gray, (new_w, new_h))

    return gray

def find_game_elements(source_img, templates, threshold=0.8):
    """
    Find multiple game elements in a screenshot.

    Args:
        source_img: Path to or numpy array of source image
        templates: Dict of template names to image paths/arrays
        threshold: Minimum confidence for matches

    Returns:
        Dict of element names to list of (location, confidence) tuples
    """
    results = {}

    for element_name, template_path in templates.items():
        matches = multi_template_match(source_img, template_path, threshold)
        results[element_name] = matches

    return results

# Example usage for Flappy Bird AI
if __name__ == "__main__":
    # Load the images from the workspace
    source_img = 'Screenshot_2025.12.28_16.09.30.581.png'  # Game screenshot
    
    # Use pipe images as templates
    templates = {
        'pipe': 'pipe.png',
        'bird': 'bird.png'
    }

    try:
        # Load source image to check dimensions
        source = cv2.imread(source_img)
        if source is None:
            print(f"Error: Could not load source image {source_img}")
            exit(1)

        print(f"Source image shape: {source.shape}")

        # Load and check template images
        for name, path in templates.items():
            template = cv2.imread(path)
            if template is None:
                print(f"Error: Could not load template image {path}")
                exit(1)
            print(f"{name} template shape: {template.shape}")

        # Find all pipe elements in the screenshot
        game_elements = find_game_elements(source_img, templates, threshold=0.7)
        
        print("\nDetected game elements:")
        for element_name, matches in game_elements.items():
            if matches:
                print(f"\n{element_name}:")
                for i, (loc, conf) in enumerate(matches[:3]):  # Show top 3 matches
                    print(f"  Match {i+1}: location {loc}, confidence {conf:.3f}")
            else:
                print(f"\n{element_name}: No matches found above threshold")

        # Label the pipes with different colors
        labeled_image = source.copy()
        
        # Colors for different pipe types
        colors = {
            'pipe_up': (0, 255, 0),    # Green for upward pipes
            'pipe_down': (0, 0, 255)   # Red for downward pipes
        }
        
        # Get template dimensions for drawing rectangles
        template_dims = {}
        for name, path in templates.items():
            template = cv2.imread(path)
            h, w = template.shape[:2]
            template_dims[name] = (w, h)
        
        # Draw rectangles around detected pipes
        for element_name, matches in game_elements.items():
            color = colors.get(element_name, (255, 255, 255))  # White as default
            w, h = template_dims[element_name]
            
            for loc, conf in matches:
                if conf >= 0.7:  # Only label high confidence matches
                    top_left = loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(labeled_image, top_left, bottom_right, color, 2)
                    
                    # Add label text
                    label = f"{element_name} ({conf:.2f})"
                    cv2.putText(labeled_image, label, (top_left[0], top_left[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save the labeled image
        output_path = 'labeled_pipes.png'
        cv2.imwrite(output_path, labeled_image)
        print(f"\nLabeled image saved as: {output_path}")
        
        # Display the image (optional - may not work in all environments)
        # cv2.imshow('Detected Pipes', labeled_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Example of how you might use this for Flappy Bird AI
        print("\nFor Flappy Bird AI, you would typically:")
        print("1. Take screenshots of just the game area")
        print("2. Use pipe templates to detect obstacles")
        print("3. Calculate distances between pipes and bird")
        print("4. Make jump decisions based on pipe positions")

    except Exception as e:
        print(f"Error processing images: {e}")
        print("Make sure all images exist and are valid image files")

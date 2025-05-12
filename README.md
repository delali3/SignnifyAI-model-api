✅ Call This from Your Mobile App
📱 Example: React Native / Flutter
Send the camera frame to your API using multipart/form-data.

```javascript

const formData = new FormData();
formData.append('file', {
  uri: fileUri,
  type: 'image/jpeg',
  name: 'frame.jpg',
});

fetch("https://your-api-url.com/predict", {
  method: 'POST',
  body: formData,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
})
  .then(res => res.json())
  .then(data => console.log(data))
  .catch(err => console.error(err));

```
✅ Call This from Your Web App
🌐 Example: React / Vue / Angular
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
fetch("https://your-api-url.com/predict", {
  method: 'POST',
  body: formData,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
})
  .then(res => res.json())
  .then(data => console.log(data))
  .catch(err => console.error(err));
```
import axios from 'axios';

// src/services/api.js
const API = axios.create({
    baseURL: 'http://localhost:8000', // 👈 directly use your FastAPI backend URL
  });
  
export const fetchSatelliteTIFFAndPredict = async (cityName) => {
  const response = await API.post('/get-satellite-image', { city: cityName });
  const fileBlob = new Blob([response.data.image], { type: 'image/tiff' });

  const formData = new FormData();
  formData.append('file', fileBlob, `${cityName}.tiff`);

  const prediction = await API.post('/predict/', formData);
  return prediction.data;
};

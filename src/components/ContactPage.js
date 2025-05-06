import React, { useState } from 'react';
import Navbar from './NavBar';
import Copyright from './CopyRight';
import '../styles/ContactPage.css';

const ContactPage = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  
  const [submitted, setSubmitted] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Here you would typically send the form data to your backend
    console.log('Form submitted:', formData);
    // Show success message
    setSubmitted(true);
    // Reset form
    setFormData({
      name: '',
      email: '',
      subject: '',
      message: ''
    });
    // Reset the success message after 5 seconds
    setTimeout(() => {
      setSubmitted(false);
    }, 5000);
  };

  return (
    <div className="contact-page-wrapper">
      <Navbar />
      <div className="feature-content container">
        <h1>Contact Us</h1>
        
        <section className="contact-info-section">
          <div className="row">
            <div className="col col-md-6">
              <div className="contact-info">
                <h2>Get in Touch</h2>
                <p>Have questions about our deepfake detection technology? Want to learn more about our services? Reach out to us!</p>
                
                <div className="contact-details">
                  <div className="contact-item">
                    <h3>Email</h3>
                    <p>info@deepfakedetection.tech</p>
                  </div>
                  
                  <div className="contact-item">
                    <h3>Phone</h3>
                    <p>+91 123 456 7890</p>
                  </div>
                  
                  <div className="contact-item">
                    <h3>Address</h3>
                    <p>
                      Department of Artificial Intelligence and Data Science<br />
                      KGiSL Institute Of Technology(Autonomous)<br />
                      Coimbatore, Tamil Nadu<br />
                      India
                    </p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="col col-md-6">
              <div className="contact-form-container">
                <h2>Send Us a Message</h2>
                {submitted && (
                  <div className="form-success-message">
                    Thank you for your message! We'll get back to you soon.
                  </div>
                )}
                <form onSubmit={handleSubmit} className="contact-form">
                  <div className="form-group">
                    <label htmlFor="name">Name</label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleChange}
                      required
                    />
                  </div>
                  
                  <div className="form-group">
                    <label htmlFor="email">Email</label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      required
                    />
                  </div>
                  
                  <div className="form-group">
                    <label htmlFor="subject">Subject</label>
                    <input
                      type="text"
                      id="subject"
                      name="subject"
                      value={formData.subject}
                      onChange={handleChange}
                      required
                    />
                  </div>
                  
                  <div className="form-group">
                    <label htmlFor="message">Message</label>
                    <textarea
                      id="message"
                      name="message"
                      rows="5"
                      value={formData.message}
                      onChange={handleChange}
                      required
                    ></textarea>
                  </div>
                  
                  <button type="submit" className="submit-btn">Send Message</button>
                </form>
              </div>
            </div>
          </div>
        </section>

        <section className="faq-section">
          <h2>Frequently Asked Questions</h2>
          <div className="faq-container">
            <div className="faq-item">
              <h3>How accurate is your deepfake detection?</h3>
              <p>Our detection technology achieves over 95% accuracy in identifying synthetic audio, image, and video content using advanced machine learning algorithms.</p>
            </div>
            
            <div className="faq-item">
              <h3>What file formats do you support?</h3>
              <p>We support all common audio formats (MP3, WAV, AAC), image formats (JPG, PNG), and video formats (MP4, MOV).</p>
            </div>
            
            <div className="faq-item">
              <h3>Can I integrate your technology with my existing systems?</h3>
              <p>Yes, we offer API integration options for enterprise clients to incorporate our detection technology into their workflows.</p>
            </div>
          </div>
        </section>
      </div>
      
      <Copyright />
    </div>
  );
};

export default ContactPage;
import React from 'react';
import Navbar from './NavBar';
import Copyright from './CopyRight';
import '../styles/Aboutpage.css';

const About = () => {
  const teamMembers = [
    {
      name: 'Ms. Akilandeswari M',
      role: 'Project Mentor & Assistant Professor',
      bio: '',
      image: '/api/placeholder/300/300'
    },
    {
      name: 'Yuvan Velkumar S',
      role: 'Student',
      bio: '',
      image: '/api/placeholder/300/300'
    },
    {
      name: 'Vishal S K',
      role: 'Student',
      bio: '',
      image: '/api/placeholder/300/300'
    },
    {
      name: 'Navin M',
      role: 'Student',
      bio: '',
      image: '/api/placeholder/300/300'
    },
    {
      name: 'Sujith C M',
      role: 'Student',
      bio: '',
      image: '/api/placeholder/300/300'
    }
  ];

  return (
    <div className="audio-deepfake-wrapper">
      <Navbar />
      <div className="feature-content">
        <h1>About Our Deepfake Detection Technology</h1>
        
        <section className="about-section">
          <p>
            Our mission is to combat the growing challenge of deepfakes by providing cutting-edge detection technology. 
            We leverage advanced machine learning algorithms to identify synthetic audio, image and video with high accuracy and reliability.
          </p>
        </section>

        <section className="mission-section">
          <h2>Our Vision</h2>
          <p>
            In an era of rapidly evolving digital technologies, we aim to protect individuals and organizations 
            from the potential misuse of AI-generated audio content. Our solution provides a robust defense 
            against audio, image and video manipulation and misinformation.
          </p>
        </section>

        <section className="team-section">
          <h2>Our Expert Team</h2>
          <div className="row">
            {teamMembers.map((member, index) => (
              <div key={index} className="col col-md-4 col-lg-3">
                <div className="team-member">
                  <div className="team-member-image-wrapper">
                    <img 
                      src={member.image} 
                      alt={member.name} 
                      className="team-member-image"
                    />
                  </div>
                  <h3>{member.name}</h3>
                  <h4>{member.role}</h4>
                  <p>{member.bio}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="technology-section">
          <h2>Our Technology</h2>
          <div className="row">
            <div className="col col-md-6">
              <div className="tech-detail">
                <h3>Advanced Detection</h3>
                <p>
                  Our AI models are trained on extensive datasets, e acoustic features 
                  to distinguish between authentic and synthesized audio with unprecedented accuracy.
                </p>
              </div>
            </div>
            <div className="col col-md-6">
              <div className="tech-detail">
                <h3>Continuous Learning</h3>
                <p>
                  We continuously update our models to stay ahead of emerging deepfake technologies, 
                  ensuring our detection remains cutting-edge and reliable.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
      <Copyright />
    </div>
  );
};

export default About;
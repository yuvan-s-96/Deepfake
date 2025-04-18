import React, { useState } from 'react';
import { Menubar } from 'primereact/menubar';
import { InputText } from 'primereact/inputtext';
import { Badge } from 'primereact/badge';
import { Avatar } from 'primereact/avatar';  
import { Link } from 'react-router-dom'; 
import '../styles/NavBar.css';
import '../styles/responsive-styles.css';
import logo from '../assets/android-chrome-512x512.png';

export default function ResponsiveNavbar() {
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    const toggleMobileMenu = () => {
        setIsMobileMenuOpen(!isMobileMenuOpen);
    };

    const itemRenderer = (item) => (
        <Link 
            to={item.to} 
            className="navbar-item-link"
            onClick={() => setIsMobileMenuOpen(false)}
        >
            {item.icon && <span className={`navbar-item-icon ${item.icon}`} />}
            <span className="navbar-item-label">{item.label}</span>
            {item.badge && <Badge className="navbar-badge" value={item.badge} />}
        </Link>
    );

    const items = [
        {
            label: 'Home',
            icon: 'pi pi-home',
            to: '/',
            template: itemRenderer
        },
        {
            label: 'About',
            icon: 'pi pi-star',
            to: '/about',
            template: itemRenderer
        },
        {
            label: 'Features',
            icon: 'pi pi-search',
            items: [
                {
                    label: 'Video Deepfake',
                    icon: 'pi pi-bolt',
                    to: '/features/video-deepfake',
                    template: itemRenderer
                },
                {
                    label: 'Image Deepfake',
                    icon: 'pi pi-server',
                    to: '/features/image-deepfake',
                    template: itemRenderer
                },
                {
                    label: 'Audio Deepfake',
                    icon: 'pi pi-pencil',
                    to: '/features/audio-deepfake',
                    template: itemRenderer
                }
            ]
        },
        {
            label: 'Contact',
            icon: 'pi pi-envelope',
            to: '/contact',
            // template: itemRenderer
        }
    ];

    const start = (
        <Link to="/" className="navbar-logo-link">
            <img 
                alt="logo" 
                src={logo} 
                className="navbar-logo"
            />
        </Link>
    );

    const end = (
        <div className="navbar-end-container">
            <div className="navbar-search-container">
                <InputText 
                    placeholder="Search" 
                    type="text" 
                    className="navbar-search-input" 
                />
            </div>
            <Avatar 
                image="https://primefaces.org/cdn/primereact/images/avatar/amyelsner.png" 
                shape="circle" 
                className="navbar-avatar" 
            />
            <button 
                className="mobile-menu-toggle"
                onClick={toggleMobileMenu}
            >
                <span className={`menu-icon ${isMobileMenuOpen ? 'open' : ''}`}></span>
            </button>
        </div>
    );

    return (
        <nav className="navbar-wrapper">
            <Menubar 
                model={items} 
                start={start} 
                end={end} 
                className={`navbar ${isMobileMenuOpen ? 'mobile-menu-open' : ''}`} 
            />
            {isMobileMenuOpen && (
                <div className="mobile-menu">
                    {items.map((item, index) => (
                        <div key={index} className="mobile-menu-item">
                            <Link 
                                to={item.to} 
                                onClick={toggleMobileMenu}
                            >
                                {item.label}
                            </Link>
                            {item.items && item.items.map((subItem, subIndex) => (
                                <Link 
                                    key={subIndex} 
                                    to={subItem.to} 
                                    className="mobile-submenu-item"
                                    onClick={toggleMobileMenu}
                                >
                                    {subItem.label}
                                </Link>
                            ))}
                        </div>
                    ))}
                </div>
            )}
        </nav>
    );
}
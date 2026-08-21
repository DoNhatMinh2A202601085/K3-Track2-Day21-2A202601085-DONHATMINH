terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-1"
}

variable "bucket_name" {
  description = "S3 bucket name for MLOps artifacts"
  type        = string
  default     = "wine-mlops-donhatminh"
}

variable "instance_type" {
  description = "EC2 instance type (Free Tier for ap-southeast-1)"
  type        = string
  default     = "t3.micro"
}

# 1. Tu dong tao cap SSH Key bang Terraform
resource "tls_private_key" "deploy_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "mlops_key" {
  key_name   = "mlops-deploy-key"
  public_key = tls_private_key.deploy_key.public_key_openssh
}

resource "local_file" "private_key" {
  content         = tls_private_key.deploy_key.private_key_pem
  filename        = "${path.module}/mlops_deploy.pem"
  file_permission = "0600"
}

# 2. Lay AMI Ubuntu 22.04 LTS moi nhat
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# 3. Tao Security Group mo cong 22 (SSH) va cong 8000 (FastAPI API)
resource "aws_security_group" "mlops_sg" {
  name        = "mlops-serve-sg"
  description = "Allow SSH and Port 8000 for MLOps FastAPI Serving"

  ingress {
    description = "SSH from anywhere"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "FastAPI Inference API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name = "mlops-serve-sg"
  }
}

# 4. Khoi tao EC2 Instance voi User Data tu dong cai dat moi truong
resource "aws_instance" "mlops_serve" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.mlops_key.key_name
  vpc_security_group_ids      = [aws_security_group.mlops_sg.id]
  associate_public_ip_address = true

  user_data = <<-EOF
              #!/bin/bash
              sudo apt update -y
              sudo apt install -y python3-pip
              pip3 install fastapi uvicorn scikit-learn joblib boto3

              mkdir -p /home/ubuntu/models /home/ubuntu/src
              chown -R ubuntu:ubuntu /home/ubuntu/models /home/ubuntu/src

              # Tao systemd service cho FastAPI
              cat <<'SERVICE' > /etc/systemd/system/mlops-serve.service
              [Unit]
              Description=MLOps Model Inference Server
              After=network.target

              [Service]
              User=ubuntu
              WorkingDirectory=/home/ubuntu
              Environment="CLOUD_BUCKET=${var.bucket_name}"
              Environment="AWS_DEFAULT_REGION=${var.aws_region}"
              ExecStart=/usr/bin/python3 /home/ubuntu/src/serve.py
              Restart=always
              RestartSec=5

              [Install]
              WantedBy=multi-user.target
              SERVICE

              systemctl daemon-reload
              systemctl enable mlops-serve
              EOF

  tags = {
    Name = "mlops-serve"
  }
}

# Outputs
output "ec2_public_ip" {
  description = "Public IP cua EC2 Instance (dien vao VM_HOST tren GitHub Secrets)"
  value       = aws_instance.mlops_serve.public_ip
}

output "ec2_user" {
  description = "User SSH (dien vao VM_USER tren GitHub Secrets)"
  value       = "ubuntu"
}

output "ssh_private_key_path" {
  description = "Duong dan file Private Key duoc luu tren may"
  value       = "${path.module}/mlops_deploy.pem"
}

output "ssh_command" {
  description = "Lenh SSH vao VM"
  value       = "ssh -i terraform/mlops_deploy.pem ubuntu@${aws_instance.mlops_serve.public_ip}"
}
